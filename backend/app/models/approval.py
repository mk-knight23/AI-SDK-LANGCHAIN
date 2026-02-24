"""Models for approval-related API operations."""
from pydantic import BaseModel, Field
from typing import Literal, Optional


class ApprovalRequest(BaseModel):
    """Request for human approval."""

    approval_type: Literal["cost", "action", "output", "critical", "normal"] = Field(
        ...,
        description="Type of approval being requested",
    )
    agent_output: str = Field(..., description="Output requiring approval")
    context: Optional[dict] = Field(None, description="Additional context")
    timeout: int = Field(default=300, description="Timeout in seconds")


class ApprovalDecision(BaseModel):
    """Decision on an approval request."""

    decision: Literal["approved", "rejected"] = Field(
        ...,
        description="Approval decision",
    )
    feedback: Optional[str] = Field(None, description="Optional feedback for rejection")


class ApprovalResponse(BaseModel):
    """Response for approval operations."""

    approval_id: str
    status: Literal["pending", "approved", "rejected", "timeout"]
    decision: Optional[Literal["approved", "rejected"]] = None
    feedback: Optional[str] = None
    created_at: Optional[str] = None
    approved_at: Optional[str] = None
    rejected_at: Optional[str] = None
