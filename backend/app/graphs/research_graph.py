"""LangGraph research agent workflow.

Performs web-based research with iterative refinement.
"""
from typing import TypedDict, Annotated, NotRequired, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.llm.providers import create_llm_with_fallback, ProviderError
from app.config import get_config

import operator


class ResearchState(TypedDict):
    """State for the research agent.

    Attributes:
        messages: List of messages
        query: Research query
        findings: Accumulated research findings
        sources: List of sources cited
        current_step: Current step in workflow
        iteration: Current iteration number
        needs_more_search: Whether more searching is needed
        summary: Final summary of findings
        error: Error message if something failed
    """

    messages: Annotated[list, operator.add]
    query: str
    findings: list[str]
    sources: list[dict]
    current_step: str
    iteration: NotRequired[int]
    needs_more_search: NotRequired[bool]
    summary: NotRequired[str]
    error: NotRequired[str]


class ResearchGraph:
    """LangGraph agent for web-based research.

    Performs iterative research with source gathering and summarization.
    Supports cyclic flows for deeper investigation.
    """

    def __init__(self, provider: str | None = None):
        """Initialize the research graph.

        Args:
            provider: LLM provider to use
        """
        config = get_config()
        self._provider = provider or config.default_llm_provider
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the research workflow graph."""
        builder = StateGraph(ResearchState)

        # Add nodes
        builder.add_node("plan_search", self._plan_search)
        builder.add_node("execute_search", self._execute_search)
        builder.add_node("analyze_findings", self._analyze_findings)
        builder.add_node("check_completion", self._check_completion)
        builder.add_node("summarize", self._summarize)

        # Set entry point
        builder.set_entry_point("plan_search")

        # Define edges
        builder.add_edge("plan_search", "execute_search")
        builder.add_edge("execute_search", "analyze_findings")
        builder.add_edge("analyze_findings", "check_completion")
        builder.add_conditional_edges(
            "check_completion",
            self._should_continue,
            {
                "continue": "plan_search",
                "summarize": "summarize",
            },
        )
        builder.add_edge("summarize", END)

        return builder.compile(checkpointer=MemorySaver())

    def _plan_search(self, state: ResearchState) -> ResearchState:
        """Plan the next search query based on findings so far.

        Args:
            state: Current research state

        Returns:
            Updated state with search plan
        """
        query = state["query"]
        findings = state.get("findings", [])
        iteration = state.get("iteration", 0)

        if iteration == 0:
            search_query = query
        else:
            # Refine search based on existing findings
            search_query = f"{query} (expanding on: {findings[-1][:100]})"

        return {
            **state,
            "current_step": "search_planned",
            "messages": [{"role": "system", "content": f"Searching for: {search_query}"}],
        }

    def _execute_search(self, state: ResearchState) -> ResearchState:
        """Execute the search and gather sources.

        Note: This is a placeholder for actual web search integration.
        In production, integrate with search APIs (Perplexity, Tavily, etc.).

        Args:
            state: Current research state

        Returns:
            Updated state with new findings
        """
        query = state["query"]
        iteration = state.get("iteration", 0)

        # Simulate search results (replace with actual search API)
        prompt = f"""Provide 3-5 key findings about: {query}

For each finding, provide:
1. A concise summary of the information
2. The type of source (academic, news, industry report, etc.)

Format as a numbered list with sources in brackets."""

        try:
            response = create_llm_with_fallback(self._provider, prompt)

            # Parse the response into findings
            findings = state.get("findings", [])
            sources = state.get("sources", [])

            # Simulate extracting findings (in production, parse properly)
            new_finding = response.content[:500]
            findings.append(new_finding)
            sources.append({"iteration": iteration, "query": query})

            return {
                **state,
                "findings": findings,
                "sources": sources,
                "current_step": "search_complete",
                "messages": [{"role": "assistant", "content": new_finding}],
            }
        except ProviderError as e:
            return {
                **state,
                "error": f"Search failed: {str(e)}",
                "current_step": "error",
            }

    def _analyze_findings(self, state: ResearchState) -> ResearchState:
        """Analyze the findings and determine if more research is needed.

        Args:
            state: Current research state

        Returns:
            Updated state with analysis
        """
        findings = state.get("findings", [])
        iteration = state.get("iteration", 0)

        # Determine if we have enough information
        has_enough = len(findings) >= 3 or iteration >= 3

        return {
            **state,
            "needs_more_search": not has_enough,
            "current_step": "findings_analyzed",
        }

    def _check_completion(self, state: ResearchState) -> ResearchState:
        """Check if research is complete or needs more iterations.

        Args:
            state: Current research state

        Returns:
            Updated state
        """
        iteration = state.get("iteration", 0)
        needs_more = state.get("needs_more_search", False)

        if not needs_more or iteration >= 3:
            return {**state, "current_step": "ready_to_summarize"}

        return {**state, "iteration": iteration + 1, "current_step": "needs_more"}

    def _should_continue(self, state: ResearchState) -> str:
        """Determine whether to continue searching or summarize.

        Args:
            state: Current research state

        Returns:
            "continue" or "summarize"
        """
        if state.get("needs_more_search", False):
            return "continue"
        return "summarize"

    def _summarize(self, state: ResearchState) -> ResearchState:
        """Create a final summary of all research findings.

        Args:
            state: Current research state

        Returns:
            Updated state with summary
        """
        findings = state.get("findings", [])
        query = state["query"]

        findings_text = "\n\n".join([f"- {f}" for f in findings])

        prompt = f"""Synthesize the following research findings about "{query}" into a comprehensive summary:

{findings_text}

Provide:
1. Executive Summary (3-4 sentences)
2. Key Findings (bullet points)
3. Important Trends or Patterns
4. Gaps in Information (if any)
5. Suggested Next Steps (if applicable)"""

        try:
            response = create_llm_with_fallback(self._provider, prompt)

            return {
                **state,
                "summary": response.content,
                "current_step": "complete",
                "messages": [{"role": "assistant", "content": response.content}],
            }
        except ProviderError as e:
            return {
                **state,
                "error": f"Summarization failed: {str(e)}",
                "current_step": "error",
            }

    def invoke(self, state: ResearchState, config: dict | None = None) -> ResearchState:
        """Invoke the research graph.

        Args:
            state: Initial research state
            config: Optional run configuration

        Returns:
            Final research state
        """
        if "iteration" not in state:
            state = {**state, "iteration": 0}
        if "findings" not in state:
            state = {**state, "findings": []}
        if "sources" not in state:
            state = {**state, "sources": []}

        return self._graph.invoke(state, config)

    def stream(self, state: ResearchState, config: dict | None = None):
        """Stream research execution.

        Args:
            state: Initial research state
            config: Optional run configuration

        Yields:
            State updates as they occur
        """
        if "iteration" not in state:
            state = {**state, "iteration": 0}
        if "findings" not in state:
            state = {**state, "findings": []}
        if "sources" not in state:
            state = {**state, "sources": []}

        for event in self._graph.stream(state, config):
            yield event


# Create default instance
default_graph = ResearchGraph()
graph = default_graph._graph
