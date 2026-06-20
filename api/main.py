import os
from collections.abc import Iterable
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent

from api.rate_limit import check_rate_limit
from deep_research_agent.graph import build_graph
from deep_research_agent.state import initial_state
from eval.dataset import QA_PAIRS

app = FastAPI(title="Deep Research Agent Demo API")

_allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# 复用同一个编译好的图：跟 main.py/eval/run_eval.py 一样的接入方式，
# nodes.py/graph.py 完全不需要为了接 API 改一行。
_graph = build_graph()

RateLimitDep = Annotated[None, Depends(check_rate_limit)]


@app.get("/api/examples")
def list_examples() -> list[str]:
    return [qa["question"] for qa in QA_PAIRS[:4]]


def _stream_research(query: str) -> Iterable[ServerSentEvent]:
    state = initial_state(query)

    # stream_mode="updates" 给的是每个节点返回的原始增量，不是合并后的全量
    # （比如 search 节点本轮新查到的结果，不包含之前几轮已经累积的）。
    # search_results 在 state.py 里用 operator.add 累积，这里手动复刻同样的语义，
    # 保证流结束时拿到的 state 跟 graph.invoke() 的返回值等价。
    for step in _graph.stream(state, stream_mode="updates"):
        for node_name, update in step.items():
            if "search_results" in update:
                state["search_results"] = state["search_results"] + update["search_results"]
            else:
                state.update(update)
            yield ServerSentEvent(data={"node": node_name, "update": update}, event="node")

    # 故意不发 state["final_report"]：那是 output_node 为 CLI 拼好的单个字符串
    # （报告 + 文本化来源列表 + 反思记录粘在一起），前端要分开渲染来源卡片和反思区块，
    # 直接发还没拼接前的几个字段更直接，不用反过来从一段文本里解析。
    yield ServerSentEvent(
        data={
            "report": state["draft_report"],
            "sources": state["search_results"],
            "critique": state["critique"],
            "iteration": state["iteration"],
        },
        event="done",
    )


@app.get("/api/research", response_class=EventSourceResponse)
def stream_research(
    query: Annotated[str, Query(min_length=1, max_length=500)],
    _: RateLimitDep,
) -> Iterable[ServerSentEvent]:
    return _stream_research(query)
