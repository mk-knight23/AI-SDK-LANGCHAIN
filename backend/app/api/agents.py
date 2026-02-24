"""API endpoints for agent operations."""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Literal

from app.models.agent import AgentRequest, AgentResponse, AgentListResponse, AgentType, AgentInfo
from app.graphs.venture_graph import VentureGraph, VentureState
from app.graphs.research_graph import ResearchGraph, ResearchState
from app.websocket.streaming import get_stream_manager, stream_agent_execution
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])

# Agent registry
_agent_registry = {
    AgentType.VENTURE: VentureGraph,
    AgentType.RESEARCH: ResearchGraph,
}


@router.get("/", response_model=AgentListResponse)
async def list_agents():
    """List all available agents.

    Returns information about available agent types including
    their capabilities and input requirements.
    """
    agents = [
        AgentInfo(
            agent_type=AgentType.VENTURE,
            name="Venture Planning Agent",
            description="Analyzes startup ideas and generates business plans, pitch decks",
            input_schema={
                "type": "object",
                "properties": {
                    "idea": {"type": "string", "description": "Startup idea to analyze"},
                },
                "required": ["idea"],
            },
            capabilities=["market_analysis", "business_model", "pitch_deck"],
        ),
        AgentInfo(
            agent_type=AgentType.RESEARCH,
            name="Research Agent",
            description="Performs web-based research with iterative refinement",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Research query"},
                },
                "required": ["query"],
            },
            capabilities=["web_search", "source_gathering", "summarization"],
        ),
    ]

    return AgentListResponse(agents=agents)


@router.post("/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """Execute an agent synchronously.

    Args:
        request: Agent execution request

    Returns:
        Agent execution result

    Raises:
        HTTPException: If agent type is invalid or execution fails
    """
    # Validate agent type
    agent_class = _agent_registry.get(request.agent_type)
    if agent_class is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent type: {request.agent_type}",
        )

    try:
        # Create agent instance
        agent = agent_class()

        # Prepare initial state based on agent type
        if request.agent_type == AgentType.VENTURE:
            initial_state: VentureState = {
                "messages": [],
                "idea": request.input.get("idea", ""),
                "market_analysis": "",
                "business_model": "",
                "pitch_deck": "",
                "current_step": "start",
            }
        elif request.agent_type == AgentType.RESEARCH:
            initial_state: ResearchState = {
                "messages": [],
                "query": request.input.get("query", ""),
                "findings": [],
                "sources": [],
                "current_step": "start",
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Agent type not fully implemented: {request.agent_type}",
            )

        # Execute agent
        result = agent.invoke(initial_state, config=request.config)

        return AgentResponse(
            agent_type=request.agent_type,
            thread_id=request.config.get("thread_id", "") if request.config else "",
            status="complete",
            result=result,
            steps_completed=result.get("iteration", 1),
        )

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}",
        ) from e


@router.websocket("/stream")
async def stream_agent(websocket: WebSocket):
    """WebSocket endpoint for streaming agent execution.

    Connect with WebSocket to receive real-time updates during
    agent execution including tokens and step transitions.

    Message format:
    ```json
    {
        "agent_type": "venture",
        "input": {"idea": "startup idea"},
        "thread_id": "optional-thread-id"
    }
    ```
    """
    await websocket.accept()
    stream_manager = get_stream_manager()

    try:
        # Receive initial request
        data = await websocket.receive_json()
        agent_type = data.get("agent_type")
        input_data = data.get("input", {})
        thread_id = data.get("thread_id", "default")

        # Validate agent type
        agent_class = _agent_registry.get(agent_type)
        if agent_class is None:
            await websocket.send_json({
                "type": "error",
                "data": {"error": f"Unknown agent type: {agent_type}"},
            })
            await websocket.close()
            return

        # Add connection
        stream_manager.add_connection(thread_id, websocket)

        # Create agent and prepare state
        agent = agent_class()

        if agent_type == AgentType.VENTURE:
            initial_state: VentureState = {
                "messages": [],
                "idea": input_data.get("idea", ""),
                "market_analysis": "",
                "business_model": "",
                "pitch_deck": "",
                "current_step": "start",
            }
        elif agent_type == AgentType.RESEARCH:
            initial_state: ResearchState = {
                "messages": [],
                "query": input_data.get("query", ""),
                "findings": [],
                "sources": [],
                "current_step": "start",
            }

        # Execute with streaming
        result = await stream_agent_execution(
            agent._graph,
            initial_state,
            thread_id,
            stream_manager,
        )

        # Send final result
        await websocket.send_json({
            "type": "complete",
            "data": {"result": result},
        })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for thread {thread_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "data": {"error": str(e)},
        })
    finally:
        stream_manager.remove_connection(thread_id, websocket)


@router.get("/types")
async def get_agent_types():
    """Get list of available agent types.

    Simple endpoint for frontend to enumerate agent options.
    """
    return {
        "types": [t.value for t in AgentType],
        "default": AgentType.VENTURE.value,
    }
