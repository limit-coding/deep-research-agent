import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import rate_limit
from api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    rate_limit._ip_request_times.clear()
    rate_limit._daily_counts.clear()
    yield
    rate_limit._ip_request_times.clear()
    rate_limit._daily_counts.clear()


def _parse_sse(text: str) -> list[tuple[str | None, str]]:
    """把 SSE 文本切成 (event, data) 列表，对应前端 EventSource 实际拿到的事件粒度。"""
    events = []
    event_name = None
    for line in text.splitlines():
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            events.append((event_name, line.removeprefix("data:").strip()))
    return events


def test_list_examples_returns_four_questions():
    response = client.get("/api/examples")
    assert response.status_code == 200
    assert len(response.json()) == 4


@patch("deep_research_agent.nodes.web_search")
@patch("deep_research_agent.nodes.get_structured_llm")
@patch("deep_research_agent.nodes.get_llm")
def test_stream_research_emits_node_events_then_done(mock_get_llm, mock_get_structured_llm, mock_web_search):
    """跟 test_graph.py 的 mock 方式完全一致（mock 掉 get_llm/get_structured_llm/web_search），

    这里只验证 API 层把 graph.stream() 的输出正确转成 SSE 事件序列，不重复验证图内部逻辑。
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

    response = client.get("/api/research", params={"query": "LangGraph 和 LangChain 有什么区别？"})

    assert response.status_code == 200
    events = _parse_sse(response.text)

    parsed_nodes = [json.loads(data)["node"] for event, data in events if event == "node"]
    # 一轮就 sufficient=True，不触发 reflect -> search 的二次循环
    assert parsed_nodes == ["decompose", "search", "synthesize", "reflect", "output"]

    done_payloads = [json.loads(data) for event, data in events if event == "done"]
    assert len(done_payloads) == 1
    assert "综合报告草稿" in done_payloads[0]["report"]
    assert done_payloads[0]["sources"] == mock_web_search.return_value
    assert done_payloads[0]["critique"] == "信息充分"


def test_rate_limit_returns_429_after_per_ip_max(monkeypatch):
    monkeypatch.setattr(rate_limit, "PER_IP_MAX", 2)
    monkeypatch.setattr(rate_limit, "DAILY_MAX", 1000)

    with (
        patch("deep_research_agent.nodes.web_search", return_value=[]),
        patch("deep_research_agent.nodes.get_structured_llm") as mock_structured,
        patch("deep_research_agent.nodes.get_llm") as mock_llm,
    ):
        mock_structured.return_value = MagicMock(
            invoke=MagicMock(
                return_value=SimpleNamespace(
                    sub_questions=["q"], sufficient=True, critique="ok", follow_up_questions=[]
                )
            )
        )
        mock_llm.return_value = MagicMock(invoke=MagicMock(return_value=SimpleNamespace(content="report")))

        assert client.get("/api/research", params={"query": "a"}).status_code == 200
        assert client.get("/api/research", params={"query": "b"}).status_code == 200
        assert client.get("/api/research", params={"query": "c"}).status_code == 429
