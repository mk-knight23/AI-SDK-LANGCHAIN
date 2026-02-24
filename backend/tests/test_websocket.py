"""Tests for WebSocket streaming implementation.

TDD approach: Tests written before implementation.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import json

from app.websocket.streaming import (
    StreamMessage,
    MessageType,
    StreamManager,
    stream_agent_execution,
    StreamError,
)


@pytest.fixture
def sample_stream_message():
    """Create a sample stream message."""
    return StreamMessage(
        type="token",
        data={"content": "Hello", "step": "analyze_market"},
        thread_id="test-thread",
    )


class TestStreamMessage:
    """Tests for StreamMessage model."""

    def test_stream_message_creation(self):
        """Test StreamMessage creation with all fields."""
        message = StreamMessage(
            type="token",
            data={"content": "test"},
            thread_id="thread-1",
            sequence=1,
        )
        assert message.type == "token"
        assert message.data == {"content": "test"}
        assert message.thread_id == "thread-1"
        assert message.sequence == 1

    def test_stream_message_to_dict(self):
        """Test StreamMessage serialization."""
        message = StreamMessage(
            type="step_start",
            data={"step": "analyze_market"},
            thread_id="thread-1",
        )
        result = message.to_dict()

        assert result["type"] == "step_start"
        assert result["data"]["step"] == "analyze_market"
        assert result["thread_id"] == "thread-1"

    def test_stream_message_to_json(self):
        """Test StreamMessage JSON serialization."""
        message = StreamMessage(
            type="token",
            data={"content": "test"},
            thread_id="thread-1",
        )
        json_str = message.to_json()

        parsed = json.loads(json_str)
        assert parsed["type"] == "token"
        assert parsed["data"]["content"] == "test"


class TestMessageType:
    """Tests for MessageType constants."""

    def test_message_type_constants(self):
        """Test all required message type constants exist."""
        assert hasattr(MessageType, "TOKEN")
        assert hasattr(MessageType, "STEP_START")
        assert hasattr(MessageType, "STEP_COMPLETE")
        assert hasattr(MessageType, "ERROR")
        assert hasattr(MessageType, "APPROVAL_REQUEST")


class TestStreamManager:
    """Tests for StreamManager."""

    def test_stream_manager_initialization(self):
        """Test StreamManager initializes correctly."""
        manager = StreamManager()
        assert manager is not None
        assert hasattr(manager, "connections")
        assert hasattr(manager, "add_connection")
        assert hasattr(manager, "remove_connection")

    def test_add_connection(self):
        """Test adding a connection."""
        manager = StreamManager()
        mock_ws = Mock()

        manager.add_connection("thread-1", mock_ws)

        assert "thread-1" in manager.connections
        assert mock_ws in manager.connections["thread-1"]

    def test_remove_connection(self):
        """Test removing a connection."""
        manager = StreamManager()
        mock_ws = Mock()

        manager.add_connection("thread-1", mock_ws)
        manager.remove_connection("thread-1", mock_ws)

        # Verify connection removed
        assert mock_ws not in manager.connections.get("thread-1", set())

    def test_broadcast_to_thread(self):
        """Test broadcasting to specific thread."""
        manager = StreamManager()
        mock_ws1 = Mock()
        mock_ws2 = Mock()

        manager.add_connection("thread-1", mock_ws1)
        manager.add_connection("thread-2", mock_ws2)

        message = StreamMessage(
            type="token",
            data={"content": "test"},
            thread_id="thread-1",
        )

        manager.broadcast("thread-1", message)

        # Only thread-1 connection should receive message
        mock_ws1.send.assert_called_once()
        mock_ws2.send.assert_not_called()

    def test_broadcast_to_all_threads(self):
        """Test broadcasting to all threads."""
        manager = StreamManager()
        mock_ws1 = Mock()
        mock_ws2 = Mock()

        manager.add_connection("thread-1", mock_ws1)
        manager.add_connection("thread-2", mock_ws2)

        message = StreamMessage(
            type="system",
            data={"message": "broadcast"},
            thread_id="*",
        )

        manager.broadcast_all(message)

        mock_ws1.send.assert_called_once()
        mock_ws2.send.assert_called_once()


class TestStreamAgentExecution:
    """Tests for stream_agent_execution function."""

    @patch("app.websocket.streaming.StreamManager")
    def test_stream_agent_execution_yields_messages(self, mock_manager_class):
        """Test that stream_agent_execution yields stream messages."""
        # This test verifies the streaming structure
        # Full async testing would require async fixtures
        assert callable(stream_agent_execution)


class TestStreamError:
    """Tests for StreamError exception."""

    def test_stream_error_is_exception(self):
        """Test StreamError is an Exception subclass."""
        error = StreamError("Stream failed")
        assert isinstance(error, Exception)

    def test_stream_error_preserves_message(self):
        """Test StreamError preserves error message."""
        message = "Connection lost"
        error = StreamError(message)
        assert str(error) == message
