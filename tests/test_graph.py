from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from deep_research_agent.graph import build_graph
from deep_research_agent.nodes import Critique, SubQuestions
from deep_research_agent.state import initial_state


def test_graph_compiles():
    assert build_graph() is not None


@patch("deep_research_agent.nodes.web_search")
@patch("deep_research_agent.nodes.get_structured_llm")
@patch("deep_research_agent.nodes.get_llm")
def test_graph_runs_end_to_end_with_mocked_llm_and_search(mock_get_llm, mock_get_structured_llm, mock_web_search):
    """图的节点连线/状态流转是否正确，不依赖真实 LLM 或搜索 API key。

    mock 掉三个唯一会发网络请求的边界函数（get_llm、get_structured_llm、web_search），
    其余完全跑真实的 LangGraph 执行逻辑，包括 reflect -> search 的循环分支。
    """
    mock_web_search.return_value = [
        {"query": "q", "title": "标题", "url": "https://example.com", "snippet": "摘要"}
    ]

    def fake_get_structured_llm(schema):
        fake_llm = MagicMock()
        if schema is SubQuestions:
            fake_llm.invoke.return_value = SimpleNamespace(sub_questions=["子问题1", "子问题2"])
        elif schema is Critique:
            # 第一次反思说不充分（触发重新搜索），第二次说充分（结束循环）
            fake_llm.invoke.side_effect = [
                SimpleNamespace(sufficient=False, critique="缺少最新数据", follow_up_questions=["补充问题"]),
                SimpleNamespace(sufficient=True, critique="信息充分", follow_up_questions=[]),
            ]
        return fake_llm

    mock_get_structured_llm.side_effect = fake_get_structured_llm
    mock_get_llm.return_value = MagicMock(invoke=MagicMock(return_value=SimpleNamespace(content="综合报告草稿")))

    graph = build_graph()
    result = graph.invoke(initial_state("LangGraph 和 LangChain 有什么区别？"))

    assert result["iteration"] == 2
    assert "综合报告草稿" in result["final_report"]
    assert "参考来源" in result["final_report"]
    # 两轮 search（初次 2 个子问题 + 反思后 1 个 follow-up）应该都被累积下来
    assert len(result["search_results"]) == 3

    # 首轮 2 次调用走 basic + 默认结果数；reflect 判定不足后的 follow-up 调用
    # 才升级到 advanced + 更多结果——验证“贵的检索预算只花在已知缺口”这个设计
    calls = mock_web_search.call_args_list
    assert len(calls) == 3
    for call in calls[:2]:
        assert call.kwargs["search_depth"] == "basic"
        assert call.kwargs["max_results"] == 4
    assert calls[2].kwargs["search_depth"] == "advanced"
    assert calls[2].kwargs["max_results"] == 8
