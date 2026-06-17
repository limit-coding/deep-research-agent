import os
import sys

from dotenv import load_dotenv

load_dotenv()

from deep_research_agent.graph import build_graph
from deep_research_agent.state import initial_state


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else input("请输入研究问题: ")

    config = {}
    # 只有配置了 Langfuse key 才接入 tracing，本地没配 key 也能跑通整条流程
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from langfuse.langchain import CallbackHandler

        config["callbacks"] = [CallbackHandler()]

    graph = build_graph()
    result = graph.invoke(initial_state(query), config=config)

    print("\n" + "=" * 40 + "\n")
    print(result["final_report"])


if __name__ == "__main__":
    main()
