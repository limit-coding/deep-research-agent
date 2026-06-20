import operator
from typing import Annotated, TypedDict


class SearchResult(TypedDict):
    query: str
    title: str
    url: str
    snippet: str

"""
把它想成一个"在5个节点之间传来传去的背包"，每个节点往里加
  点东西。唯一不直观的地方是 search_results 那个
  Annotated[..., operator.add]——LangGraph
  默认规则是"节点返回什么字段，就覆盖背包里原来那个字段"，但
  搜索这个节点会被进两次（第一次查初始子问题，反思后可能再查
  一次），如果是覆盖，第一轮搜到的资料就被冲掉了。加这个标记
  的意思是"这个字段遇到新值不覆盖，而是拼接"。
"""
class ResearchState(TypedDict):
    query: str
    sub_questions: list[str]
    # 用 operator.add 累积而不是覆盖：反思后会回到 search 再查一轮，
    # 综合/输出阶段需要看到全部历史来源，不只是最后一轮的结果。
    search_results: Annotated[list[SearchResult], operator.add]
    draft_report: str
    critique: str
    needs_more_research: bool
    follow_up_questions: list[str]
    iteration: int
    final_report: str


def initial_state(query: str) -> ResearchState:
    return {
        "query": query,
        "sub_questions": [],
        "search_results": [],
        "draft_report": "",
        "critique": "",
        "needs_more_research": False,
        "follow_up_questions": [],
        "iteration": 0,
        "final_report": "",
    }
