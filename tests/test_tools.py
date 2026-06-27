from unittest.mock import MagicMock, patch

import deep_research_agent.tools as tools_module
from deep_research_agent.tools import web_search


def test_web_search_retries_on_transient_failure_then_succeeds():
    """瞬时失败（网络抖动/429）后重试成功，最终返回结果而不是空列表。"""
    good_response = {"results": [{"title": "标题", "url": "https://example.com", "content": "内容"}]}
    mock_client = MagicMock()
    # 前两次抛异常，第三次成功
    mock_client.search.side_effect = [ConnectionError("timeout"), ConnectionError("timeout"), good_response]

    # patch time.sleep 让 tenacity 的退避等待变成零延迟，否则测试要等 1s + 2s
    with patch.object(tools_module, "_get_client", return_value=mock_client), patch("time.sleep"):
        results = web_search("LangGraph 是什么")

    assert len(results) == 1
    assert results[0]["url"] == "https://example.com"
    assert mock_client.search.call_count == 3


def test_web_search_returns_empty_list_after_all_retries_exhausted():
    """3 次重试全部失败后，返回空列表而不是抛出异常，不让整条 agent 流程崩掉。"""
    mock_client = MagicMock()
    mock_client.search.side_effect = ConnectionError("network unreachable")

    with patch.object(tools_module, "_get_client", return_value=mock_client), patch("time.sleep"):
        results = web_search("某个查询")

    assert results == []
    assert mock_client.search.call_count == 3
