from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from eval.run_eval import (
    JudgeScore,
    ReportQualityScore,
    accuracy_evaluator,
    citation_count_evaluator,
    citation_support_evaluator,
    report_quality_evaluator,
    research_task,
)


@patch("deep_research_agent.nodes.web_search")
@patch("deep_research_agent.nodes.get_structured_llm")
@patch("deep_research_agent.nodes.get_llm")
def test_research_task_runs_graph_and_returns_report_and_sources(
    mock_get_llm, mock_get_structured_llm, mock_web_search
):
    """research_task 是 eval 和图之间的唯一接口：给一个有 .input 的 item，

    拿到 {"final_report", "sources"}。mock 掉的几个函数跟 test_graph.py 一致，
    这里只验证 research_task 自己的包装逻辑，不重复验证图内部的连线。
    """
    mock_web_search.return_value = [
        {"query": "q", "title": "标题", "url": "https://example.com", "snippet": "摘要"}
    ]

    def fake_get_structured_llm(schema):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = SimpleNamespace(
            sub_questions=["子问题1"],
            sufficient=True,
            critique="信息充分",
            follow_up_questions=[],
        )
        return fake_llm

    mock_get_structured_llm.side_effect = fake_get_structured_llm
    mock_get_llm.return_value = MagicMock(invoke=MagicMock(return_value=SimpleNamespace(content="综合报告草稿")))

    item = SimpleNamespace(input="LangGraph 和 LangChain 有什么区别？")
    result = research_task(item=item)

    assert "综合报告草稿" in result["final_report"]
    assert result["sources"] == mock_web_search.return_value


@patch("eval.run_eval.get_structured_llm")
def test_accuracy_evaluator_wraps_judge_score_into_evaluation(mock_get_structured_llm):
    fake_judge_llm = MagicMock()
    fake_judge_llm.invoke.return_value = JudgeScore(score=0.9, reasoning="覆盖了标准答案要点")
    mock_get_structured_llm.return_value = fake_judge_llm

    evaluation = accuracy_evaluator(
        input="问题", output={"final_report": "报告正文"}, expected_output="标准答案"
    )

    assert evaluation.name == "accuracy"
    assert evaluation.value == 0.9
    assert evaluation.comment == "覆盖了标准答案要点"


@patch("eval.run_eval.get_structured_llm")
def test_citation_support_evaluator_handles_empty_sources(mock_get_structured_llm):
    fake_judge_llm = MagicMock()
    fake_judge_llm.invoke.return_value = JudgeScore(score=0.0, reasoning="没有任何来源，无法支持结论")
    mock_get_structured_llm.return_value = fake_judge_llm

    evaluation = citation_support_evaluator(
        input="问题", output={"final_report": "报告正文 [1]", "sources": []}
    )

    assert evaluation.name == "citation_support"
    assert evaluation.value == 0.0


def test_citation_count_evaluator_counts_distinct_valid_citations_without_llm_call():
    """不依赖 judge：[1][1][2][99] 在只有 2 条来源时，应该只数出 {1, 2} 两条——

    重复引用不重复计数，超出来源范围的编号（大概率是模型瞎编的）不计入。
    """
    output = {"final_report": "结论A [1]。结论B [1][2]。结论C [99]。", "sources": [{}, {}]}

    evaluation = citation_count_evaluator(input="问题", output=output)

    assert evaluation.name == "citation_count"
    assert evaluation.value == 2


@patch("eval.run_eval.get_structured_llm")
def test_report_quality_evaluator_returns_three_evaluations_from_one_judge_call(mock_get_structured_llm):
    fake_judge_llm = MagicMock()
    fake_judge_llm.invoke.return_value = ReportQualityScore(
        comprehensiveness=0.8, depth=0.6, readability=0.9, reasoning="覆盖主要维度但分析较浅"
    )
    mock_get_structured_llm.return_value = fake_judge_llm

    evaluations = report_quality_evaluator(input="问题", output={"final_report": "报告正文"})

    assert mock_get_structured_llm.call_count == 1
    names = {e.name: e.value for e in evaluations}
    assert names == {"comprehensiveness": 0.8, "depth": 0.6, "readability": 0.9}
