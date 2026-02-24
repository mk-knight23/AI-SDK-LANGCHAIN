"""LangGraph approval gate workflow.

Implements human-in-the-loop approval gates for agent workflows.
"""
from typing import TypedDict, Annotated, NotRequired, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import operator
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ApprovalState(TypedDict):
    """State for the approval workflow.

    Attributes:
        messages: List of messages
        agent_output: Output from the agent requiring approval
        approval_id: Unique identifier for this approval request
        approval_type: Type of approval (cost, action, output, etc.)
        context: Additional context for the approval
        decision: User decision (pending/approved/rejected)
        feedback: User feedback if rejected
        approved_at: Timestamp when approved
        rejected_at: Timestamp when rejected
        timeout: Timeout for approval (seconds)
        created_at: Timestamp when created
        error: Error message if something failed
    """

    messages: Annotated[list, operator.add]
    agent_output: str
    approval_id: str
    approval_type: Literal["cost", "action", "output", "critical", "normal"]
    context: NotRequired[dict]
    decision: NotRequired[Literal["pending", "approved", "rejected"]]
    feedback: NotRequired[str]
    approved_at: NotRequired[str]
    rejected_at: NotRequired[str]
    timeout: NotRequired[int]
    created_at: NotRequired[str]
    error: NotRequired[str]


class ApprovalGraph:
    """LangGraph workflow for human-in-the-loop approval gates.

    Manages approval requests with timeout handling and decision tracking.
    Can be integrated into other workflows as a subprocess.
    """

    def __init__(self):
        """Initialize the approval graph."""
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the approval workflow graph."""
        builder = StateGraph(ApprovalState)

        # Add nodes
        builder.add_node("create_request", self._create_request)
        builder.add_node("wait_for_approval", self._wait_for_approval)
        builder.add_node("check_timeout", self._check_timeout)
        builder.add_node("process_decision", self._process_decision)

        # Set entry point
        builder.set_entry_point("create_request")

        # Define edges
        builder.add_edge("create_request", "wait_for_approval")
        builder.add_conditional_edges(
            "wait_for_approval",
            self._has_decision,
            {
                "decision": "process_decision",
                "timeout": "check_timeout",
            },
        )
        builder.add_edge("check_timeout", "process_decision")
        builder.add_edge("process_decision", END)

        return builder.compile(checkpointer=MemorySaver())

    def _create_request(self, state: ApprovalState) -> ApprovalState:
        """Create the approval request.

        Args:
            state: Current approval state

        Returns:
            Updated state with request created
        """
        return {
            **state,
            "decision": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "current_step": "request_created",
            "messages": [{"role": "system", "content": "Approval request created"}],
        }

    def _wait_for_approval(self, state: ApprovalState) -> ApprovalState:
        """Wait for human approval decision.

        In a real implementation, this would:
        1. Send notification via WebSocket
        2. Wait for incoming decision
        3. Handle timeout

        For this implementation, we check if decision has been made.

        Args:
            state: Current approval state

        Returns:
            Updated state
        """
        decision = state.get("decision", "pending")
        current_step = "waiting_approval"

        if decision != "pending":
            current_step = "decision_received"

        return {**state, "current_step": current_step}

    def _has_decision(self, state: ApprovalState) -> str:
        """Check if a decision has been made.

        Args:
            state: Current approval state

        Returns:
            "decision" or "timeout"
        """
        decision = state.get("decision", "pending")

        if decision == "pending":
            # Check timeout
            created_at = state.get("created_at")
            timeout = state.get("timeout", 300)  # 5 minutes default

            if created_at:
                created = datetime.fromisoformat(created_at)
                elapsed = (datetime.utcnow() - created).total_seconds()

                if elapsed > timeout:
                    return "timeout"

        return "decision"

    def _check_timeout(self, state: ApprovalState) -> ApprovalState:
        """Handle approval timeout.

        Args:
            state: Current approval state

        Returns:
            Updated state with timeout decision
        """
        logger.warning(f"Approval {state.get('approval_id')} timed out")

        return {
            **state,
            "decision": "rejected",
            "feedback": "Approval timed out",
            "rejected_at": datetime.utcnow().isoformat(),
            "current_step": "timeout",
        }

    def _process_decision(self, state: ApprovalState) -> ApprovalState:
        """Process the approval decision.

        Args:
            state: Current approval state

        Returns:
            Final approval state
        """
        decision = state.get("decision", "pending")

        if decision == "approved":
            return {
                **state,
                "approved_at": datetime.utcnow().isoformat(),
                "current_step": "approved",
                "messages": [{"role": "system", "content": "Approval granted"}],
            }
        elif decision == "rejected":
            return {
                **state,
                "rejected_at": datetime.utcnow().isoformat(),
                "current_step": "rejected",
                "messages": [{"role": "system", "content": "Approval denied"}],
            }
        else:
            return {
                **state,
                "current_step": "pending",
            }

    def invoke(self, state: ApprovalState, config: dict | None = None) -> ApprovalState:
        """Invoke the approval workflow.

        Args:
            state: Initial approval state
            config: Optional run configuration

        Returns:
            Final approval state
        """
        return self._graph.invoke(state, config)

    def stream(self, state: ApprovalState, config: dict | None = None):
        """Stream approval workflow execution.

        Args:
            state: Initial approval state
            config: Optional run configuration

        Yields:
            State updates as they occur
        """
        for event in self._graph.stream(state, config):
            yield event

    def approve(self, approval_id: str) -> ApprovalState:
        """Approve an approval request.

        Args:
            approval_id: The approval request ID

        Returns:
            Updated state with approval
        """
        return {
            "approval_id": approval_id,
            "decision": "approved",
            "approved_at": datetime.utcnow().isoformat(),
        }

    def reject(self, approval_id: str, feedback: str | None = None) -> ApprovalState:
        """Reject an approval request.

        Args:
            approval_id: The approval request ID
            feedback: Optional rejection feedback

        Returns:
            Updated state with rejection
        """
        state = {
            "approval_id": approval_id,
            "decision": "rejected",
            "rejected_at": datetime.utcnow().isoformat(),
        }

        if feedback:
            state["feedback"] = feedback

        return state


# Create default instance
default_graph = ApprovalGraph()
graph = default_graph._graph
