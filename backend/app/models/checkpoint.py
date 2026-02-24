"""Models for checkpoint-related API operations."""
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


class CheckpointCreate(BaseModel):
    """Request to create a checkpoint."""

    thread_id: str = Field(..., description="Thread identifier")
    state: dict[str, Any] = Field(..., description="Agent state to checkpoint")
    step: int = Field(default=0, description="Step number")
    parent_id: Optional[str] = Field(None, description="Parent checkpoint ID")
    metadata: Optional[dict[str, Any]] = Field(None, description="Checkpoint metadata")


class CheckpointResponse(BaseModel):
    """Response with checkpoint data."""

    thread_id: str
    checkpoint_id: str
    state: dict[str, Any]
    step: int
    parent_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime


class CheckpointListResponse(BaseModel):
    """Response listing checkpoints for a thread."""

    thread_id: str
    checkpoints: list[CheckpointResponse]
    total: int
