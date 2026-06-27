from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from deep_research_agent.graph import build_graph
from deep_research_agent.nodes import Critique, SubQuestions, search_node
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

    每次 web_search 返回唯一 URL，以便同时验证跨轮去重逻辑。
    """
    # 每次 web_search 调用返回一个不同 URL，模拟真实的跨查询不重叠来源
    mock_web_search.side_effect = [
        [{"query": "q", "title": f"标题{i}", "url": f"https://example{i}.com", "snippet": "摘要"}]
        for i in range(10)
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
    # 两轮 search 共 3 次调用，每次返回不同 URL，全部保留
    assert len(result["search_results"]) == 3

    # 首轮 2 次调用走 basic + 默认结果数；reflect 判定不足后的 follow-up 调用
    # 才升级到 advanced + 更多结果——验证"贵的检索预算只花在已知缺口"这个设计
    calls = mock_web_search.call_args_list
    assert len(calls) == 3
    for call_args in calls[:2]:
        assert call_args.kwargs["search_depth"] == "basic"
        assert call_args.kwargs["max_results"] == 4
    assert calls[2].kwargs["search_depth"] == "advanced"
    assert calls[2].kwargs["max_results"] == 8


@patch("deep_research_agent.nodes.web_search")
def test_search_node_deduplicates_results_by_url(mock_web_search):
    """同一 URL 在不同轮次的搜索里出现时，第二次出现应该被过滤掉。

    这防止 Tavily 对相关查询返回同一批来源时，synthesize prompt 里堆重复内容。
    """
    duplicate_url = "https://docs.example.com/api"
    mock_web_search.return_value = [
        {"query": "q", "title": "文档", "url": duplicate_url, "snippet": "内容"}
    ]

    # 模拟状态：第一轮已经把这个 URL 查到了
    state = {
        "sub_questions": ["子问题1"],
        "follow_up_questions": ["补充问题"],
        "search_results": [{"query": "q", "title": "文档", "url": duplicate_url, "snippet": "内容"}],
        "iteration": 1,
    }

    result = search_node(state)

    # 同一 URL 不应该再次出现在返回的新结果里
    assert result["search_results"] == []


@patch("deep_research_agent.nodes.web_search")
def test_search_node_keeps_new_urls_from_followup(mock_web_search):
    """follow-up 轮查到的全新 URL 应该正常加入。"""
    mock_web_search.return_value = [
        {"query": "q", "title": "新来源", "url": "https://new-source.com", "snippet": "新内容"}
    ]

    state = {
        "sub_questions": ["子问题1"],
        "follow_up_questions": ["补充问题"],
        "search_results": [{"query": "q", "title": "旧来源", "url": "https://old-source.com", "snippet": "旧内容"}],
        "iteration": 1,
    }

    result = search_node(state)

    assert len(result["search_results"]) == 1
    assert result["search_results"][0]["url"] == "https://new-source.com"
