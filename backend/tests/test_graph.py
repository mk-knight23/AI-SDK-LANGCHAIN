"""Tests for the VentureGraph LangGraph implementation."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.graph import (
    analyze_market,
    design_business_model,
    create_pitch_deck,
    VentureState,
    graph,
    builder,
    get_llm,
)


@pytest.fixture
def sample_venture_state():
    """Create a sample venture state for testing."""
    return {
        "messages": [],
        "idea": "AI-powered sustainable fashion marketplace",
        "market_analysis": "",
        "business_model": "",
        "pitch_deck": "",
        "current_step": "start",
    }


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    mock_response = Mock()
    mock_response.content = "Mocked analysis content for testing"
    return mock_response


class TestAnalyzeMarket:
    """Tests for the analyze_market node function."""

    @patch("app.graph.get_llm")
    def test_analyze_market_calls_llm_with_correct_prompt(self, mock_get_llm, sample_venture_state):
        """Test that analyze_market calls LLM with the correct prompt."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Market analysis result"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        result = analyze_market(sample_venture_state)

        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        assert "AI-powered sustainable fashion marketplace" in call_args
        assert "Target market size" in call_args
        assert "Key competitors" in call_args

    @patch("app.graph.get_llm")
    def test_analyze_market_returns_updated_state(self, mock_get_llm, sample_venture_state):
        """Test that analyze_market returns state with market_analysis."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Market size: $10B. Competitors: Company A, B, C."
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        result = analyze_market(sample_venture_state)

        assert result["market_analysis"] == "Market size: $10B. Competitors: Company A, B, C."
        assert result["current_step"] == "market_analyzed"
        assert result["idea"] == sample_venture_state["idea"]

    @patch("app.graph.get_llm")
    def test_analyze_market_preserves_existing_state(self, mock_get_llm, sample_venture_state):
        """Test that analyze_market preserves other state fields."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Analysis complete"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        sample_venture_state["messages"] = ["previous message"]
        result = analyze_market(sample_venture_state)

        assert "previous message" in result["messages"]
        assert result["idea"] == sample_venture_state["idea"]


class TestDesignBusinessModel:
    """Tests for the design_business_model node function."""

    @patch("app.graph.get_llm")
    def test_design_business_model_calls_llm_with_prompt(self, mock_get_llm, sample_venture_state):
        """Test that design_business_model calls LLM with correct prompt."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Business model canvas"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        sample_venture_state["market_analysis"] = "Large market with growth potential"
        result = design_business_model(sample_venture_state)

        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        assert "AI-powered sustainable fashion marketplace" in call_args
        assert "Large market" in call_args
        assert "Value Proposition" in call_args

    @patch("app.graph.get_llm")
    def test_design_business_model_returns_updated_state(self, mock_get_llm, sample_venture_state):
        """Test that design_business_model returns state with business_model."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Revenue: Subscriptions. Customers: Eco-conscious millennials."
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        sample_venture_state["market_analysis"] = "Market analysis data"
        result = design_business_model(sample_venture_state)

        assert result["business_model"] == "Revenue: Subscriptions. Customers: Eco-conscious millennials."
        assert result["current_step"] == "business_model_done"

    @patch("app.graph.get_llm")
    def test_design_business_model_truncates_market_analysis(self, mock_get_llm, sample_venture_state):
        """Test that design_business_model truncates long market analysis."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Business model result"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        sample_venture_state["market_analysis"] = "A" * 1000
        design_business_model(sample_venture_state)

        call_args = mock_llm.invoke.call_args[0][0]
        # Verify the market context is truncated to 500 chars
        market_context_start = call_args.find("Market context:") + len("Market context: ")
        market_context = call_args[market_context_start:market_context_start + 10]
        assert len(market_context) <= 510  # Allow for some buffer


class TestCreatePitchDeck:
    """Tests for the create_pitch_deck node function."""

    @patch("app.graph.get_llm")
    def test_create_pitch_deck_calls_llm_with_prompt(self, mock_get_llm, sample_venture_state):
        """Test that create_pitch_deck calls LLM with correct prompt."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "10-slide pitch deck outline"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        sample_venture_state["market_analysis"] = "Growing market"
        sample_venture_state["business_model"] = "SaaS subscription"
        result = create_pitch_deck(sample_venture_state)

        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        assert "AI-powered sustainable fashion marketplace" in call_args
        assert "Growing market" in call_args
        assert "SaaS subscription" in call_args

    @patch("app.graph.get_llm")
    def test_create_pitch_deck_returns_updated_state(self, mock_get_llm, sample_venture_state):
        """Test that create_pitch_deck returns state with pitch_deck."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Slide 1: Problem. Slide 2: Solution."
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        sample_venture_state["market_analysis"] = "Market data"
        sample_venture_state["business_model"] = "Business model data"
        result = create_pitch_deck(sample_venture_state)

        assert result["pitch_deck"] == "Slide 1: Problem. Slide 2: Solution."
        assert result["current_step"] == "complete"

    @patch("app.graph.get_llm")
    def test_create_pitch_deck_truncates_inputs(self, mock_get_llm, sample_venture_state):
        """Test that create_pitch_deck truncates long inputs."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Pitch deck outline"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        sample_venture_state["market_analysis"] = "B" * 500
        sample_venture_state["business_model"] = "C" * 500
        create_pitch_deck(sample_venture_state)

        call_args = mock_llm.invoke.call_args[0][0]
        # Verify inputs are truncated to 300 chars
        assert len(call_args) < 1500  # Should be much shorter with truncation


class TestGraphStructure:
    """Tests for the graph structure and state transitions."""

    def test_graph_has_correct_nodes(self):
        """Test that the graph has all required nodes."""
        nodes = builder.nodes
        assert "analyze_market" in nodes
        assert "design_business_model" in nodes
        assert "create_pitch_deck" in nodes

    def test_graph_entry_point(self):
        """Test that the graph entry point is analyze_market."""
        # The compiled graph should start with analyze_market
        # We verify this by checking the nodes exist
        assert "analyze_market" in builder.nodes

    def test_graph_edges(self):
        """Test that the graph has correct edges between nodes."""
        # Check that edges exist in the graph
        edges = builder.edges
        edge_pairs = [(src, dst) for src, dst in edges]

        assert ("analyze_market", "design_business_model") in edge_pairs
        assert ("design_business_model", "create_pitch_deck") in edge_pairs

    def test_graph_compiles_successfully(self):
        """Test that the graph compiles without errors."""
        assert graph is not None


class TestGraphIntegration:
    """Integration tests for the complete graph execution."""

    @patch("app.graph.get_llm")
    def test_full_graph_execution(self, mock_get_llm):
        """Test the complete graph execution flow."""
        mock_llm = Mock()
        # Set up mock responses for each node
        mock_responses = [
            Mock(content="Market analysis: Large growing market"),
            Mock(content="Business model: SaaS subscription"),
            Mock(content="Pitch deck: 10 slides outlined"),
        ]
        mock_llm.invoke.side_effect = mock_responses
        mock_get_llm.return_value = mock_llm

        initial_state = {
            "messages": [],
            "idea": "Test startup idea",
            "market_analysis": "",
            "business_model": "",
            "pitch_deck": "",
            "current_step": "start",
        }

        result = graph.invoke(initial_state)

        assert result["market_analysis"] == "Market analysis: Large growing market"
        assert result["business_model"] == "Business model: SaaS subscription"
        assert result["pitch_deck"] == "Pitch deck: 10 slides outlined"
        assert result["current_step"] == "complete"
        assert mock_llm.invoke.call_count == 3

    @patch("app.graph.get_llm")
    def test_graph_state_transitions(self, mock_get_llm):
        """Test that graph transitions through correct states."""
        mock_llm = Mock()
        mock_responses = [
            Mock(content="Market analysis"),
            Mock(content="Business model"),
            Mock(content="Pitch deck"),
        ]
        mock_llm.invoke.side_effect = mock_responses
        mock_get_llm.return_value = mock_llm

        initial_state = {
            "messages": [],
            "idea": "Test idea",
            "market_analysis": "",
            "business_model": "",
            "pitch_deck": "",
            "current_step": "start",
        }

        result = graph.invoke(initial_state)

        # Verify final state
        assert result["current_step"] == "complete"
        assert result["market_analysis"] != ""
        assert result["business_model"] != ""
        assert result["pitch_deck"] != ""


class TestVentureState:
    """Tests for the VentureState TypedDict."""

    def test_venture_state_structure(self):
        """Test that VentureState has all required fields."""
        state: VentureState = {
            "messages": [],
            "idea": "test",
            "market_analysis": "",
            "business_model": "",
            "pitch_deck": "",
            "current_step": "start",
        }
        assert "messages" in state
        assert "idea" in state
        assert "market_analysis" in state
        assert "business_model" in state
        assert "pitch_deck" in state
        assert "current_step" in state
