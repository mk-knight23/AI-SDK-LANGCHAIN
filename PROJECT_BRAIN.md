# Project Brain: AI-SDK-LANGCHAIN

## Purpose

Orchestration-focused chain composition for enterprise workflows.

## Current State

- Skill-aware routing through the shared core package.
- FastAPI service with normalized mission input/output.
- CLI runner for repeatable local checks.
- LangChain prompt orchestration path.
- Dockerfile, GitHub CI, pytest contract tests, and repo metadata.

## Upgrade Direction

- Add model providers, tools, and structured output parsers.
- Create reusable chain templates for common mission types.
- Add LangSmith tracing hooks and regression evals.

## Quality Bar

- Keep the repository runnable from a fresh clone.
- Keep generated caches and local secrets out of git.
- Keep README, skill matrix, tests, and CI aligned with actual behavior.
