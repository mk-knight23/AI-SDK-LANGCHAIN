"""LangGraph venture planning agent."""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import operator
import os


class VentureState(TypedDict):
    messages: Annotated[list, operator.add]
    idea: str
    market_analysis: str
    business_model: str
    pitch_deck: str
    current_step: str


# Lazy initialization of LLM to allow testing without API key
_llm = None


def get_llm():
    """Get or create the LLM instance."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    return _llm


# For testing: allow injection of mock LLM
llm = property(lambda self: get_llm())


def analyze_market(state: VentureState) -> VentureState:
    idea = state["idea"]
    prompt = f"""Analyze the market for this startup idea: {idea}
Provide: 1) Target market size, 2) Key competitors, 3) Market trends, 4) Entry barriers"""
    response = get_llm().invoke(prompt)
    return {**state, "market_analysis": response.content, "current_step": "market_analyzed"}


def design_business_model(state: VentureState) -> VentureState:
    idea = state["idea"]
    market = state["market_analysis"]
    prompt = f"""Create business model canvas for: {idea}
Market context: {market[:500]}
Provide: Value Proposition, Customer Segments, Revenue Streams, Cost Structure, Key Partnerships"""
    response = get_llm().invoke(prompt)
    return {**state, "business_model": response.content, "current_step": "business_model_done"}


def create_pitch_deck(state: VentureState) -> VentureState:
    idea = state["idea"]
    market = state["market_analysis"]
    model = state["business_model"]
    prompt = f"""Create 10-slide pitch deck outline for: {idea}
Market: {market[:300]}
Business Model: {model[:300]}
Provide slide titles and key bullet points for each."""
    response = get_llm().invoke(prompt)
    return {**state, "pitch_deck": response.content, "current_step": "complete"}


builder = StateGraph(VentureState)
builder.add_node("analyze_market", analyze_market)
builder.add_node("design_business_model", design_business_model)
builder.add_node("create_pitch_deck", create_pitch_deck)
builder.set_entry_point("analyze_market")
builder.add_edge("analyze_market", "design_business_model")
builder.add_edge("design_business_model", "create_pitch_deck")
builder.add_edge("create_pitch_deck", END)
graph = builder.compile()
