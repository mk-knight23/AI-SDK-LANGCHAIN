"""LangGraph venture planning agent workflow.

Enhanced version of the original venture graph with:
- Multi-provider LLM support
- Checkpointing capability
- Streaming support
- Approval gate integration
"""
from typing import TypedDict, Annotated, NotRequired, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.llm.providers import create_llm_with_fallback, ProviderError
from app.config import get_config

import operator


class VentureState(TypedDict):
    """State for the venture planning agent.

    Attributes:
        messages: List of messages (accumulates with operator.add)
        idea: Startup idea to analyze
        market_analysis: Analysis of the market
        business_model: Business model canvas
        pitch_deck: Generated pitch deck
        current_step: Current step in the workflow
        iteration: Current iteration number (for cyclic flows)
        needs_approval: Whether human approval is needed
        approval_status: Status of human approval (pending/approved/rejected)
        max_iterations: Maximum number of iterations to perform
        error: Error message if something failed
    """

    messages: Annotated[list, operator.add]
    idea: str
    market_analysis: str
    business_model: str
    pitch_deck: str
    current_step: str
    iteration: NotRequired[int]
    needs_approval: NotRequired[bool]
    approval_status: NotRequired[Literal["pending", "approved", "rejected"]]
    max_iterations: NotRequired[int]
    error: NotRequired[str]


class VentureGraph:
    """LangGraph agent for venture planning.

    Takes a startup idea and generates:
    1. Market analysis
    2. Business model canvas
    3. Pitch deck outline

    Supports cyclic iteration for refinement and human-in-the-loop approval.
    """

    def __init__(self, provider: str | None = None):
        """Initialize the venture graph.

        Args:
            provider: LLM provider to use (uses default if not specified)
        """
        config = get_config()
        self._provider = provider or config.default_llm_provider
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        builder = StateGraph(VentureState)

        # Add nodes
        builder.add_node("analyze_market", self._analyze_market)
        builder.add_node("design_business_model", self._design_business_model)
        builder.add_node("create_pitch_deck", self._create_pitch_deck)
        builder.add_node("check_approval", self._check_approval)
        builder.add_node("iterate_or_finish", self._iterate_or_finish)

        # Set entry point
        builder.set_entry_point("analyze_market")

        # Define edges (workflow)
        builder.add_edge("analyze_market", "design_business_model")
        builder.add_edge("design_business_model", "create_pitch_deck")
        builder.add_edge("create_pitch_deck", "check_approval")
        builder.add_conditional_edges(
            "check_approval",
            self._should_iterate,
            {
                "iterate": "analyze_market",
                "finish": END,
            },
        )

        # Compile with checkpointing
        return builder.compile(checkpointer=MemorySaver())

    def _analyze_market(self, state: VentureState) -> VentureState:
        """Analyze the market for the startup idea.

        Args:
            state: Current agent state

        Returns:
            Updated state with market analysis
        """
        idea = state["idea"]
        prompt = f"""Analyze the market for this startup idea: {idea}

Provide:
1) Target market size (TAM, SAM, SOM)
2) Key competitors and their strengths/weaknesses
3) Market trends and growth rate
4) Entry barriers and regulatory considerations

Be specific and data-driven in your analysis."""

        try:
            response = create_llm_with_fallback(self._provider, prompt)
            return {
                **state,
                "market_analysis": response.content,
                "current_step": "market_analyzed",
                "messages": [{"role": "assistant", "content": response.content}],
            }
        except ProviderError as e:
            return {
                **state,
                "error": f"Market analysis failed: {str(e)}",
                "current_step": "error",
            }

    def _design_business_model(self, state: VentureState) -> VentureState:
        """Create a business model canvas.

        Args:
            state: Current agent state

        Returns:
            Updated state with business model
        """
        idea = state["idea"]
        market = state["market_analysis"]

        prompt = f"""Create a business model canvas for: {idea}

Market context: {market[:500]}

Provide for each section:
1) Value Proposition: What problem do you solve?
2) Customer Segments: Who are your target customers?
3) Revenue Streams: How will you make money?
4) Cost Structure: What are your main costs?
5) Key Partnerships: Who do you need to work with?
6) Key Activities: What must you do?
7) Key Resources: What do you need?

Format as a structured business model canvas."""

        try:
            response = create_llm_with_fallback(self._provider, prompt)
            return {
                **state,
                "business_model": response.content,
                "current_step": "business_model_done",
                "messages": [{"role": "assistant", "content": response.content}],
            }
        except ProviderError as e:
            return {
                **state,
                "error": f"Business model design failed: {str(e)}",
                "current_step": "error",
            }

    def _create_pitch_deck(self, state: VentureState) -> VentureState:
        """Create a pitch deck outline.

        Args:
            state: Current agent state

        Returns:
            Updated state with pitch deck
        """
        idea = state["idea"]
        market = state["market_analysis"]
        model = state["business_model"]

        prompt = f"""Create a 10-slide pitch deck outline for: {idea}

Market: {market[:300]}
Business Model: {model[:300]}

For each slide provide:
- Slide Title
- 3-5 bullet points

Slides should cover:
1. Problem
2. Solution
3. Market Opportunity
4. Business Model
5. Competition/Positioning
6. Go-to-Market Strategy
7. Team (placeholder)
8. Financial Projections (placeholder)
9. Ask/Use of Funds
10. Contact/Next Steps"""

        try:
            response = create_llm_with_fallback(self._provider, prompt)
            return {
                **state,
                "pitch_deck": response.content,
                "current_step": "pitch_deck_complete",
                "messages": [{"role": "assistant", "content": response.content}],
            }
        except ProviderError as e:
            return {
                **state,
                "error": f"Pitch deck creation failed: {str(e)}",
                "current_step": "error",
            }

    def _check_approval(self, state: VentureState) -> VentureState:
        """Check if approval is needed and current status.

        Args:
            state: Current agent state

        Returns:
            State with approval information
        """
        approval_status = state.get("approval_status", "approved")
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        needs_approval = iteration < max_iterations and approval_status != "approved"

        return {
            **state,
            "needs_approval": needs_approval,
            "current_step": "approval_checked",
        }

    def _should_iterate(self, state: VentureState) -> str:
        """Determine whether to iterate or finish.

        Args:
            state: Current agent state

        Returns:
            "iterate" or "finish"
        """
        approval_status = state.get("approval_status", "approved")
        needs_approval = state.get("needs_approval", False)
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        # Iterate if approval needed and under max iterations
        if approval_status == "approved" or iteration >= max_iterations:
            return "finish"

        return "iterate"

    def invoke(self, state: VentureState, config: dict | None = None) -> VentureState:
        """Invoke the graph with initial state.

        Args:
            state: Initial state
            config: Optional configuration for the run

        Returns:
            Final state after graph execution
        """
        # Initialize iteration counter
        if "iteration" not in state:
            state = {**state, "iteration": 0}

        return self._graph.invoke(state, config)

    def stream(self, state: VentureState, config: dict | None = None):
        """Stream graph execution token by token.

        Args:
            state: Initial state
            config: Optional configuration

        Yields:
            State updates as they occur
        """
        if "iteration" not in state:
            state = {**state, "iteration": 0}

        for event in self._graph.stream(state, config):
            yield event


# Create default instance
default_graph = VentureGraph()
graph = default_graph._graph
