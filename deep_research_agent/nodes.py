from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .llm import get_llm, get_structured_llm
from .state import ResearchState
from .tools import web_search

MAX_SUB_QUESTIONS = 4
MAX_RESULTS_PER_QUERY = 4


class SubQuestions(BaseModel):
    sub_questions: list[str] = Field(description="拆解出的子问题，每个都应该可以独立用搜索引擎查到答案")


class Critique(BaseModel):
    sufficient: bool = Field(description="当前报告草稿是否信息充分、且每条结论都有对应来源支持")
    critique: str = Field(description="具体问题：哪里证据不足、哪里引用对不上结论")
    follow_up_questions: list[str] = Field(default_factory=list, description="若不充分，需要补充检索的新问题")

"""- decompose：把用户的问题拆成最多4个小问题。为什么拆？直接
  拿一整句复杂问题去搜索，召回质量差，拆成具体小问题更容易查
  到东西。"""
def decompose_node(state: ResearchState) -> dict:
    """把原始问题拆成几个独立子问题。

    直接拿整句话去搜索召回质量差；拆成具体子问题更接近人做研究时
    "分头查资料"的方式，也方便后面按子问题归因来源。
    """
    structured_llm = get_structured_llm(SubQuestions)
    result = structured_llm.invoke(
        [
            HumanMessage(
                content=(
                    f"将下面的研究问题拆解成最多 {MAX_SUB_QUESTIONS} 个具体子问题，"
                    f"每个子问题要能独立用搜索引擎查到答案。\n\n问题：{state['query']}"
                )
            )
        ]
    )
    return {"sub_questions": result.sub_questions[:MAX_SUB_QUESTIONS]}

"""- search：把当前要查的问题（第一轮是
  sub_questions，反思后是 follow_up_questions）一个个丢给
  Tavily 查，结果塞进背包。"""
def search_node(state: ResearchState) -> dict:
    """执行检索：第一轮查 sub_questions，反思后的轮次查 follow_up_questions。

    这是图里"循环/重试"真正落地的节点——reflect 的反向边带着新的
    follow_up_questions 再次进入这里，search_results 用 operator.add 累积，
    不会丢掉上一轮已经查到的来源。
    """
    queries = state.get("follow_up_questions") or state["sub_questions"]
    results = []
    for q in queries:
        results.extend(web_search(q, max_results=MAX_RESULTS_PER_QUERY))
    return {"search_results": results}

"""- synthesize：把背包里目前累积的所有搜索结果丢给
  LLM，让它写一份带引用编号 [1][2] 的报告草稿。"""
def synthesize_node(state: ResearchState) -> dict:
    """基于目前累积的全部来源，综合写一份带引用编号的报告草稿。"""
    llm = get_llm()
    sources_text = format_sources(state["search_results"])
    response = llm.invoke(
        [
            HumanMessage(
                content=(
                    f"研究问题：{state['query']}\n\n"
                    f"以下是检索到的资料（带编号），写一份结构化报告草稿。"
                    f"每个结论后面用 [编号] 标注来源，不要编造资料里没有的信息：\n\n{sources_text}"
                )
            )
        ]
    )
    return {"draft_report": response.content}

""" - reflect：让 LLM 自己批评一下刚写的草稿——有没有结论没资料
  撑着、引用编号对不对得上、是否有明显信息缺口。给出
  sufficient（够不够）和
  follow_up_questions（如果不够，还要查什么）。"""
def reflect_node(state: ResearchState) -> dict:
    """Reflexion 式自我批评：检查草稿信息是否充分、引用是否真的支持结论。

    借鉴了 Karpathy autoresearch 系列里 keep/revert 的迭代评估思路——
    每轮产出先自我评估再决定要不要继续，而不是无条件相信模型第一次的输出。
    iteration 计数配合 route_after_reflect 里的上限，避免反思循环不收敛。
    """
    structured_llm = get_structured_llm(Critique)
    sources_text = format_sources(state["search_results"])
    result = structured_llm.invoke(
        [
            HumanMessage(
                content=(
                    f"研究问题：{state['query']}\n\n报告草稿：\n{state['draft_report']}\n\n"
                    f"可用资料：\n{sources_text}\n\n"
                    "检查：1）是否有结论缺乏资料支持或引用编号对不上；2）是否有明显信息缺口。"
                )
            )
        ]
    )
    return {
        "critique": result.critique,
        "needs_more_research": not result.sufficient,
        "follow_up_questions": result.follow_up_questions,
        "iteration": state["iteration"] + 1,
    }

"""- output：草稿 + 来源列表 + 反思记录拼成最终报告。"""
def output_node(state: ResearchState) -> dict:
    """整理最终报告：草稿 + 来源列表 + 反思记录。"""
    sources_text = format_sources(state["search_results"])
    parts = [state["draft_report"], "\n\n## 参考来源\n", sources_text]
    if state.get("critique"):
        parts.append(f"\n\n## 反思记录（共 {state['iteration']} 轮）\n{state['critique']}")
    return {"final_report": "".join(parts)}


def format_sources(results: list[dict]) -> str:
    return "\n".join(f"[{i + 1}] {r['title']} - {r['snippet']} ({r['url']})" for i, r in enumerate(results))
