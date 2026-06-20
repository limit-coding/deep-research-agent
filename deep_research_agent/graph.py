import os

from langgraph.graph import END, StateGraph

from .nodes import decompose_node, output_node, reflect_node, search_node, synthesize_node
from .state import ResearchState


def route_after_reflect(state: ResearchState) -> str:
    """reflect 之后的唯一分支：回 search 重新查，还是进 output 收尾。

    LLM 的自我批评不保证收敛——如果它一直觉得"信息不足"，没有上限就会死循环，
    所以这里用 MAX_RESEARCH_ITERATIONS 做硬性退出条件，而不是只信任 needs_more_research。
    """
    max_iterations = int(os.getenv("MAX_RESEARCH_ITERATIONS", "2"))
    if state["needs_more_research"] and state["iteration"] < max_iterations:
        return "search"
    return "output"


def build_graph():
    """组装拆解 -> 搜索 -> 综合 -> 反思 -> (重搜 | 输出) 的研究流程图。

    为什么不用 LangGraph 自带的 ReAct 模式（让 LLM 自己决定何时调搜索工具）：
    这里要的是确定性的多步流程，方便单独评测每一步、预估搜索调用次数上限，
    也方便在 Langfuse trace 里按节点名对齐分析，而不是把控制权完全交给 LLM。
    """
    graph = StateGraph(ResearchState)

    graph.add_node("decompose", decompose_node)
    graph.add_node("search", search_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("output", output_node)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "search")
    graph.add_edge("search", "synthesize")
    graph.add_edge("synthesize", "reflect")
    graph.add_conditional_edges("reflect", route_after_reflect, {"search": "search", "output": "output"})
    graph.add_edge("output", END)
    """
    前四条边是固定路线，没什么好讲的。唯一的分支在 reflect
  之后：route_after_reflect 这个函数会看"是否还需要更多研究"
  + "已经反思了几轮"，决定走回 search（再查一遍）还是去
  output（收尾）。这条"回头"的边就是整张图里唯一的循环，对应
  CLAUDE.md 要求的"反思发现信息不足就重新检索"。
    """

    return graph.compile()

