"""把 QA_PAIRS 灌进 Langfuse Dataset，跑一次实验：Agent 真实研究每道题，
两个 LLM-as-judge evaluator 分别打"准确性"和"引用是否真的支持结论"的分。

用法：
    uv run python -m eval.run_eval
"""

import os

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langfuse import Evaluation, Langfuse
from pydantic import BaseModel, Field

from deep_research_agent.graph import build_graph
from deep_research_agent.llm import get_structured_llm
from deep_research_agent.nodes import format_sources
from deep_research_agent.state import initial_state
from eval.dataset import DATASET_NAME, QA_PAIRS

# 评测题本身不需要反复横跳搜索/反思，跑一次图够用，复用同一个编译好的图。
_graph = build_graph()


class JudgeScore(BaseModel):
    score: float = Field(description="0~1 之间的分数，1 表示完全符合标准，0 表示完全不符合")
    reasoning: str = Field(description="给出该分数的具体依据，引用报告或来源里的原文")


def _judge(prompt: str) -> JudgeScore:
    # judge 复用 get_structured_llm()，跟 decompose/reflect 同一套 structured-output 套路，
    # 不另起一套解析逻辑——也意味着 judge 和被测 agent 用的是同一个模型，这是已知的取舍
    # （更严谨的做法是用更强的模型当 judge，这里先保持简单，成本可控）。
    return get_structured_llm(JudgeScore).invoke([HumanMessage(content=prompt)])


def research_task(*, item, **kwargs) -> dict:
    state = _graph.invoke(initial_state(item.input))
    return {"final_report": state["final_report"], "sources": state["search_results"]}


def accuracy_evaluator(*, input, output, expected_output, metadata=None, **kwargs) -> Evaluation:
    judge = _judge(
        f"问题：{input}\n\n标准答案要点：{expected_output}\n\n"
        f"待评测报告：\n{output['final_report']}\n\n"
        "请评估待评测报告的结论是否覆盖了标准答案要点、有没有事实性错误。"
        "0~1 打分，1 表示完全准确覆盖要点且无事实错误，0 表示完全跑题或事实错误。"
    )
    return Evaluation(name="accuracy", value=judge.score, comment=judge.reasoning)


def citation_support_evaluator(*, input, output, expected_output=None, metadata=None, **kwargs) -> Evaluation:
    sources_text = format_sources(output["sources"]) or "（无检索结果）"
    judge = _judge(
        f"以下是报告引用的来源原文（按编号列出）：\n{sources_text}\n\n"
        f"报告正文（结论后用 [编号] 标注来源）：\n{output['final_report']}\n\n"
        "请检查报告里的每个引用编号是否真的能在对应来源原文里找到支持，而不是编号对不上、"
        "或者来源根本没提到这个结论。0~1 打分，1 表示所有引用都确实被来源支持。"
    )
    return Evaluation(name="citation_support", value=judge.score, comment=judge.reasoning)


def _seed_dataset(langfuse: Langfuse) -> None:
    # datasets.create 在 Langfuse 后端按 (project, name) 是 upsert 语义，重复跑不会报错；
    # dataset item 显式传 id 同理可以重复 seed 而不产生重复条目。
    langfuse.create_dataset(
        name=DATASET_NAME,
        description="Mini DeepResearch Agent 评测集：准确性 + 引用是否真的支持结论",
    )
    for qa in QA_PAIRS:
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=qa["id"],
            input=qa["question"],
            expected_output=qa["reference_answer"],
        )


def main() -> None:
    langfuse = Langfuse()
    _seed_dataset(langfuse)

    dataset = langfuse.get_dataset(DATASET_NAME)
    result = dataset.run_experiment(
        name=DATASET_NAME,
        task=research_task,
        evaluators=[accuracy_evaluator, citation_support_evaluator],
        # 每题要走完整的搜索+综合+反思流程，多题并发容易撞 Tavily/LLM 的速率限制，
        # 默认保守一点；调试单题时可以设更大的值。
        max_concurrency=int(os.getenv("EVAL_MAX_CONCURRENCY", "3")),
    )

    print(result.format(include_item_results=True))
    langfuse.flush()


if __name__ == "__main__":
    main()
