# Current Feature Inventory

## Repository

- Name: `AI-SDK-LANGCHAIN`
- SDK: LangChain
- Positioning: Orchestration-focused chain composition for enterprise workflows.

## Implemented Today

- Skill-aware routing through the shared core package.
- FastAPI service with normalized mission input/output.
- CLI runner for repeatable local checks.
- LangChain prompt orchestration path.
- Dockerfile, GitHub CI, pytest contract tests, and repo metadata.

## Not Yet Implemented

- Add model providers, tools, and structured output parsers.
- Create reusable chain templates for common mission types.
- Add LangSmith tracing hooks and regression evals.

## Verification Contract

- The local runner must complete without crashing when optional SDK credentials are missing.
- The API contract must return routing and verification fields.
- Tests must prove mission routing and a security-focused SENTINEL route.
