from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.graph import graph
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="VentureGraph - AI Venture Planning")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VentureIdea(BaseModel):
    idea: str


class VenturePlan(BaseModel):
    idea: str
    market_analysis: str
    business_model: str
    pitch_deck: str


@app.get("/health")
def health():
    return {"status": "healthy", "service": "venture-graph"}


@app.post("/plan", response_model=VenturePlan)
def create_plan(venture: VentureIdea):
    result = graph.invoke({
        "idea": venture.idea, "messages": [], "market_analysis": "",
        "business_model": "", "pitch_deck": "", "current_step": "start"
    })
    return VenturePlan(
        idea=venture.idea, market_analysis=result["market_analysis"],
        business_model=result["business_model"], pitch_deck=result["pitch_deck"]
    )
