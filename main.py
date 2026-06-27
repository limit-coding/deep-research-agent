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


def _build_langfuse_config() -> dict:
    """构造 LangGraph invoke config，只在配置了 Langfuse key 时才接入 tracing。

    用 @observe 作为顶层 span，再把 trace_context 传给 CallbackHandler，
    让 LangGraph 节点和 LLM 调用产生的 span 挂在同一个 trace 树下。
    @observe 装饰的 web_search 调用则通过 OpenTelemetry context propagation
    自动成为子 span——不需要手动传 ID。
    """
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return {}

    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    lf = get_client()
    trace_context = {
        "trace_id": lf.get_current_trace_id(),
        "parent_span_id": lf.get_current_observation_id(),
    }
    # 过滤掉 None 值：未在 @observe 内部调用时，两个 ID 均为 None，
    # 传 None 给 CallbackHandler 会触发警告。
    return {"callbacks": [CallbackHandler(trace_context={k: v for k, v in trace_context.items() if v})]}


def main():
    from langfuse import observe

    query = sys.argv[1] if len(sys.argv) > 1 else input("请输入研究问题: ")

    @observe(name="deep-research")
    def _run(q: str) -> dict:
        graph = build_graph()
        return graph.invoke(initial_state(q), config=_build_langfuse_config())

    result = _run(query)

    print("\n" + "=" * 40 + "\n")
    print(result["final_report"])


if __name__ == "__main__":
    main()
