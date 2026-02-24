"""WebSocket streaming for real-time agent execution.

Provides real-time streaming of:
- LLM tokens as they are generated
- Agent step transitions
- Approval requests
- Error notifications

Uses WebSocket for bidirectional communication with the frontend.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable
import json
import logging
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of stream messages."""

    TOKEN = "token"  # Individual LLM token
    STEP_START = "step_start"  # Agent step starting
    STEP_COMPLETE = "step_complete"  # Agent step complete
    ERROR = "error"  # Error occurred
    APPROVAL_REQUEST = "approval_request"  # Approval needed
    APPROVAL_RESPONSE = "approval_response"  # Approval decision
    STATE_UPDATE = "state_update"  # Agent state update
    COMPLETE = "complete"  # Execution complete


class StreamError(Exception):
    """Exception raised when streaming operations fail."""

    pass


@dataclass(frozen=True)
class StreamMessage:
    """A message sent over the WebSocket stream.

    Attributes:
        type: Message type from MessageType enum
        data: Message payload (varies by type)
        thread_id: Thread/conversation identifier
        sequence: Sequence number for ordering
        timestamp: Message timestamp
    """

    type: str
    data: dict[str, Any]
    thread_id: str
    sequence: int = 0
    timestamp: str = field(default_factory=lambda: "")

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "type": self.type,
            "data": self.data,
            "thread_id": self.thread_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())


class StreamManager:
    """Manages WebSocket connections and broadcasts.

    Handles connection lifecycle and message broadcasting to
    connected clients.
    """

    def __init__(self) -> None:
        """Initialize the stream manager."""
        self._connections: dict[str, set[Any]] = defaultdict(set)
        self._sequence: dict[str, int] = defaultdict(int)

    @property
    def connections(self) -> dict[str, set[Any]]:
        """Get all connections."""
        return self._connections

    def add_connection(self, thread_id: str, websocket: Any) -> None:
        """Add a WebSocket connection.

        Args:
            thread_id: Thread identifier
            websocket: WebSocket connection object
        """
        self._connections[thread_id].add(websocket)
        logger.debug(f"Added connection for thread {thread_id}")

    def remove_connection(self, thread_id: str, websocket: Any) -> None:
        """Remove a WebSocket connection.

        Args:
            thread_id: Thread identifier
            websocket: WebSocket connection object
        """
        self._connections[thread_id].discard(websocket)

        if not self._connections[thread_id]:
            del self._connections[thread_id]
            del self._sequence[thread_id]

        logger.debug(f"Removed connection for thread {thread_id}")

    async def send(self, websocket: Any, message: StreamMessage) -> bool:
        """Send a message to a specific WebSocket.

        Args:
            websocket: WebSocket connection
            message: Message to send

        Returns:
            True if sent successfully
        """
        try:
            await websocket.send_json(message.to_dict())
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    async def broadcast(self, thread_id: str, message: StreamMessage) -> int:
        """Broadcast a message to all connections for a thread.

        Args:
            thread_id: Thread identifier
            message: Message to broadcast

        Returns:
            Number of connections message was sent to
        """
        if thread_id not in self._connections:
            return 0

        # Assign sequence number
        self._sequence[thread_id] += 1
        message = StreamMessage(
            type=message.type,
            data=message.data,
            thread_id=thread_id,
            sequence=self._sequence[thread_id],
        )

        # Send to all connections for this thread
        sent_count = 0
        connections = list(self._connections[thread_id])

        for ws in connections:
            if await self.send(ws, message):
                sent_count += 1
            else:
                # Remove failed connection
                self.remove_connection(thread_id, ws)

        return sent_count

    async def broadcast_all(self, message: StreamMessage) -> int:
        """Broadcast a message to all connected threads.

        Args:
            message: Message to broadcast

        Returns:
            Total number of messages sent
        """
        total_sent = 0

        for thread_id in list(self._connections.keys()):
            sent = await self.broadcast(thread_id, message)
            total_sent += sent

        return total_sent

    def get_connection_count(self, thread_id: str) -> int:
        """Get number of connections for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Number of active connections
        """
        return len(self._connections.get(thread_id, set()))

    def get_all_thread_ids(self) -> list[str]:
        """Get all active thread IDs.

        Returns:
            List of thread IDs with active connections
        """
        return list(self._connections.keys())


# Global stream manager instance
_stream_manager = StreamManager()


def get_stream_manager() -> StreamManager:
    """Get the global stream manager instance.

    Returns:
        Global StreamManager instance
    """
    return _stream_manager


async def stream_agent_execution(
    graph,
    initial_state: dict,
    thread_id: str,
    stream_manager: StreamManager | None = None,
) -> dict:
    """Execute an agent graph with real-time streaming.

    Streams token-by-token output and state updates to connected clients.

    Args:
        graph: LangGraph agent to execute
        initial_state: Initial state for the agent
        thread_id: Thread identifier for routing
        stream_manager: Optional stream manager (uses global if not provided)

    Returns:
        Final agent state

    Raises:
        StreamError: If streaming fails
    """
    manager = stream_manager or _stream_manager

    if manager.get_connection_count(thread_id) == 0:
        logger.warning(f"No connections for thread {thread_id}, executing without streaming")

    try:
        # Send execution start message
        await manager.broadcast(
            thread_id,
            StreamMessage(
                type=MessageType.STATE_UPDATE,
                data={"status": "started", "state": initial_state},
                thread_id=thread_id,
            ),
        )

        # Stream the graph execution
        final_state = None
        step_count = 0

        for event in graph.stream(initial_state):
            step_count += 1

            # Extract node name and state
            node_name = None
            node_state = None

            for node, state in event.items():
                node_name = node
                node_state = state

            if node_name and node_state:
                # Send step start
                await manager.broadcast(
                    thread_id,
                    StreamMessage(
                        type=MessageType.STEP_START,
                        data={"step": node_name, "state": node_state},
                        thread_id=thread_id,
                    ),
                )

                # If there's content, stream it
                if "messages" in node_state:
                    for msg in node_state.get("messages", []):
                        if isinstance(msg, dict) and "content" in msg:
                            await manager.broadcast(
                                thread_id,
                                StreamMessage(
                                    type=MessageType.TOKEN,
                                    data={"content": msg["content"], "step": node_name},
                                    thread_id=thread_id,
                                ),
                            )

                # Send step complete
                await manager.broadcast(
                    thread_id,
                    StreamMessage(
                        type=MessageType.STEP_COMPLETE,
                        data={"step": node_name},
                        thread_id=thread_id,
                    ),
                )

            final_state = node_state

        # Send completion message
        await manager.broadcast(
            thread_id,
            StreamMessage(
                type=MessageType.COMPLETE,
                data={"status": "complete", "steps": step_count},
                thread_id=thread_id,
            ),
        )

        return final_state or {}

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")

        # Send error message
        await manager.broadcast(
            thread_id,
            StreamMessage(
                type=MessageType.ERROR,
                data={"error": str(e)},
                thread_id=thread_id,
            ),
        )

        raise StreamError(f"Agent execution failed: {e}") from e


async def stream_llm_response(
    provider,
    prompt: str,
    thread_id: str,
    stream_manager: StreamManager | None = None,
) -> str:
    """Stream an LLM response token by token.

    Args:
        provider: LLM provider to use
        prompt: Prompt to send
        thread_id: Thread identifier
        stream_manager: Optional stream manager

    Returns:
        Complete response text

    Raises:
        StreamError: If streaming fails
    """
    manager = stream_manager or _stream_manager

    try:
        full_response = ""

        # Stream tokens from LLM
        for token in provider.stream(prompt):
            full_response += token

            # Send each token
            await manager.broadcast(
                thread_id,
                StreamMessage(
                    type=MessageType.TOKEN,
                    data={"content": token},
                    thread_id=thread_id,
                ),
            )

        return full_response

    except Exception as e:
        logger.error(f"LLM streaming failed: {e}")

        await manager.broadcast(
            thread_id,
            StreamMessage(
                type=MessageType.ERROR,
                data={"error": str(e)},
                thread_id=thread_id,
            ),
        )

        raise StreamError(f"LLM streaming failed: {e}") from e
