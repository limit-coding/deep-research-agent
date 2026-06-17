import operator
from typing import Annotated, TypedDict


class SearchResult(TypedDict):
    query: str
    title: str
    url: str
    snippet: str


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
