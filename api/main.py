import logging
import os
from collections.abc import Iterable
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langfuse import Langfuse
from pydantic import BaseModel

from api.rate_limit import check_rate_limit
from deep_research_agent.graph import build_graph
from deep_research_agent.state import initial_state
from eval.dataset import QA_PAIRS

logger = logging.getLogger(__name__)

app = FastAPI(title="Deep Research Agent Demo API")

_allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# 复用同一个编译好的图：跟 main.py/eval/run_eval.py 一样的接入方式，
# nodes.py/graph.py 完全不需要为了接 API 改一行。
_graph = build_graph()

RateLimitDep = Annotated[None, Depends(check_rate_limit)]

BADCASE_DATASET_NAME = "badcases"


@app.get("/api/examples")
def list_examples() -> list[str]:
    return [qa["question"] for qa in QA_PAIRS[:4]]


def _build_langfuse_config() -> dict:
    """在 start_as_current_observation 上下文内调用，把当前 span 的 trace_id/span_id
    传给 CallbackHandler，让 LangGraph 节点产生的子 span 挂到同一棵 trace 树下。"""
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return {}
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    lf = get_client()
    trace_context = {
        "trace_id": lf.get_current_trace_id(),
        "parent_span_id": lf.get_current_observation_id(),
    }
    return {"callbacks": [CallbackHandler(trace_context={k: v for k, v in trace_context.items() if v})]}


def _stream_research(query: str) -> Iterable[ServerSentEvent]:
    from contextlib import nullcontext

    state = initial_state(query)

    # 有 Langfuse key 时建根 span，让 web_search/@observe 和 LangGraph CallbackHandler
    # 产生的子 span 都挂到同一棵 trace 树下；没有 key 时用 nullcontext 不做任何事。
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from langfuse import get_client

        trace_ctx = get_client().start_as_current_observation(name="deep-research")
    else:
        trace_ctx = nullcontext()

    with trace_ctx:
        # stream_mode="updates" 给的是每个节点返回的原始增量，不是合并后的全量
        # （比如 search 节点本轮新查到的结果，不包含之前几轮已经累积的）。
        # search_results 在 state.py 里用 operator.add 累积，这里手动复刻同样的语义，
        # 保证流结束时拿到的 state 跟 graph.invoke() 的返回值等价。
        for step in _graph.stream(state, config=_build_langfuse_config(), stream_mode="updates"):
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


class FeedbackRequest(BaseModel):
    query: str
    report: str
    sources: list[dict]
    rating: str  # "good" | "bad"


@app.post("/api/feedback")
def submit_feedback(body: FeedbackRequest) -> dict:
    """用户对报告的满意度反馈。

    只有 rating="bad" 的 badcase 才写入 Langfuse Dataset——thumbs-up 没有信息量，
    不值得存储。Langfuse 未配置时静默跳过，不影响前端显示。

    这条 API 关闭了 Badcase 数据进入 eval 队列的"闭环"：用户发现报告质量差 →
    badcase 自动被记录 → 下次跑 eval experiment 时可以用 langfuse.get_dataset("badcases")
    把这批题加进来，观察优化是否真的改善了这类问题。
    """
    if body.rating == "bad" and os.getenv("LANGFUSE_PUBLIC_KEY"):
        try:
            lf = Langfuse()
            lf.create_dataset(
                name=BADCASE_DATASET_NAME,
                description="用户标记为不满意的报告，待人工审核后加入评测集",
            )
            lf.create_dataset_item(
                dataset_name=BADCASE_DATASET_NAME,
                input=body.query,
                metadata={"report": body.report, "sources": body.sources},
            )
            lf.flush()
        except Exception:
            logger.warning("Failed to log badcase to Langfuse", exc_info=True)

    return {"ok": True}
