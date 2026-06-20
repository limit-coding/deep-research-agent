import os
import sys

from dotenv import load_dotenv

load_dotenv()

from deep_research_agent.graph import build_graph
from deep_research_agent.state import initial_state
"""一句话总结整个项目：一个固定路线、带一次循环的5步流水线，循环靠状态里的 iteration
  计数器兜底防止死循环，关键的"模型该输出什么格式"的环节都用 Pydantic schema
  约束，而不是裸文本解析。"""
"""
raph = build_graph()
    result = graph.invoke(initial_state(query), config=config)
两件事：build_graph()把5个节点拼成一张图（结构），initial_
  state(query)造一个初始的"状态字典"（数据）。LangGraph
  的核心模型就是：状态在节点之间传递，每个节点是一个函数，读
  状态、做点事、返回要更新的那部分状态。
  """

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
