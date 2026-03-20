import argparse

try:
    from .app import run_langchain_mission
except ImportError:
    from app import run_langchain_mission


def demo(mission: str) -> None:
    out = run_langchain_mission(mission)
    print("[LangChain] primary:", out.get("primary"))
    print("[LangChain] support:", out.get("support"))
    print("[LangChain] result:", out.get("result"))
    print("[LangChain] verification:", out.get("verification"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", default="build and secure a new API, then deploy it")
    args = parser.parse_args()
    demo(args.mission)
