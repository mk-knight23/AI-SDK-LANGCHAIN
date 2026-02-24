"""Models for agent-related API operations."""
from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum


class AgentType(str, Enum):
    """Available agent types."""

    VENTURE = "venture"
    RESEARCH = "research"
    CODE_GEN = "code_gen"


class AgentRequest(BaseModel):
    """Request to execute an agent."""

    agent_type: AgentType = Field(
        ...,
        description="Type of agent to execute",
    )
    input: dict = Field(
        ...,
        description="Input data for the agent (varies by type)",
    )
    config: Optional[dict] = Field(
        None,
        description="Optional configuration for the agent run",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the execution",
    )


class AgentResponse(BaseModel):
    """Response from agent execution."""

    agent_type: AgentType
    thread_id: str
    status: Literal["pending", "running", "complete", "error"]
    result: Optional[dict] = None
    error: Optional[str] = None
    steps_completed: int = 0


class AgentInfo(BaseModel):
    """Information about an available agent."""

    agent_type: AgentType
    name: str
    description: str
    input_schema: dict
    capabilities: list[str]


class AgentListResponse(BaseModel):
    """Response listing available agents."""

    agents: list[AgentInfo]
