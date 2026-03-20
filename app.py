"""Production-style LangChain runtime for Kazi's Agents Army."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "core"))
from agents_army_core import MissionRequest, build_mission_plan, render_system_instructions


def run_langchain_mission(mission_text: str) -> dict:
    plan = build_mission_plan(MissionRequest(mission_text))
    sys_msg = render_system_instructions(plan)

    try:
        from langchain_core.prompts import ChatPromptTemplate
    except Exception as exc:
        return {
            "primary": plan.primary,
            "support": plan.support,
            "result": None,
            "verification": f"LangChain dependency missing: {exc}",
        }

    prompt = ChatPromptTemplate.from_template("{system}\nMission: {mission}")
    rendered = prompt.format(system=sys_msg, mission=mission_text)

    return {
        "primary": plan.primary,
        "support": plan.support,
        "result": rendered,
        "verification": "LangChain prompt orchestration succeeded.",
    }
