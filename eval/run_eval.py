"""把 QA_PAIRS 灌进 Langfuse Dataset，跑一次实验：Agent 真实研究每道题，
四个 evaluator 分别打分。维度命名参考了 DeepResearch Bench（arXiv:2506.11763）
评测深度研究 agent 的两套框架——RACE（报告质量：Comprehensiveness/Depth/
Readability/Instruction Following）和 FACT（引用质量：Citation Accuracy +
effective citation count）。这里没有照搬全部维度（比如没单独评 Instruction
Following，因为它和 accuracy_evaluator 已经覆盖的"是否切题"高度重叠），
是有取舍的子集，不是论文的完整复刻。

用法：
    uv run python -m eval.run_eval
"""

import os
import re

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


class ReportQualityScore(BaseModel):
    comprehensiveness: float = Field(description="0~1，是否覆盖了这个研究问题应该涉及的关键维度，而不是只答了字面问到的一小部分")
    depth: float = Field(description="0~1，是否有分析（原因/影响/权衡取舍），而不是停留在罗列检索到的事实")
    readability: float = Field(description="0~1，结构是否清晰、表达是否通顺")
    reasoning: str = Field(description="给出以上三项打分的具体依据")


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
    """对应 DeepResearch Bench FACT 框架里的 Citation Accuracy：引用编号能不能在来源原文里找到支持。"""
    sources_text = format_sources(output["sources"]) or "（无检索结果）"
    judge = _judge(
        f"以下是报告引用的来源原文（按编号列出）：\n{sources_text}\n\n"
        f"报告正文（结论后用 [编号] 标注来源）：\n{output['final_report']}\n\n"
        "请检查报告里的每个引用编号是否真的能在对应来源原文里找到支持，而不是编号对不上、"
        "或者来源根本没提到这个结论。0~1 打分，1 表示所有引用都确实被来源支持。"
    )
    return Evaluation(name="citation_support", value=judge.score, comment=judge.reasoning)


def citation_count_evaluator(*, input, output, expected_output=None, metadata=None, **kwargs) -> Evaluation:
    """FACT 框架里 effective citation count 的简化版：不调用 LLM，直接数报告正文里

    实际出现过的、落在有效范围内的不同引用编号——citation_support 只回答"引用准不准"，
    这里回答"用了几条检索到的来源"，避免一份只引用 1 条来源、剩下证据全没用上的报告
    在 citation_support 上也能拿到高分。
    """
    total = len(output["sources"])
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", output["final_report"])}
    distinct_cited = len({n for n in cited if 1 <= n <= total})
    return Evaluation(
        name="citation_count",
        value=distinct_cited,
        comment=f"报告引用了 {distinct_cited}/{total} 条检索到的来源",
    )


def report_quality_evaluator(*, input, output, expected_output=None, metadata=None, **kwargs) -> list[Evaluation]:
    """对应 RACE 框架里的 Comprehensiveness / Depth / Readability 三维度（Instruction Following

    没有单独评，因为它和 accuracy_evaluator 已经覆盖的"是否切题"重叠）。三项揉进一次结构化
    输出而不是拆三个 judge 调用，是为了评测脚本本身的成本——15 题 x 3 次额外 LLM 调用的差距，
    比起合并成 1 次调用、用 schema 拆出 3 个字段，没有带来更多信息量。
    """
    judge = get_structured_llm(ReportQualityScore).invoke(
        [
            HumanMessage(
                content=(
                    f"研究问题：{input}\n\n报告：\n{output['final_report']}\n\n"
                    "请从三个维度评估这份研究报告：\n"
                    "1. comprehensiveness：是否覆盖了这个问题应该涉及的关键维度\n"
                    "2. depth：是否有深入分析，而不是简单罗列检索到的事实\n"
                    "3. readability：结构是否清晰、表达是否流畅\n"
                    "每项 0~1 打分。"
                )
            )
        ]
    )
    return [
        Evaluation(name="comprehensiveness", value=judge.comprehensiveness, comment=judge.reasoning),
        Evaluation(name="depth", value=judge.depth, comment=judge.reasoning),
        Evaluation(name="readability", value=judge.readability, comment=judge.reasoning),
    ]


def _seed_dataset(langfuse: Langfuse) -> None:
    # datasets.create 在 Langfuse 后端按 (project, name) 是 upsert 语义，重复跑不会报错；
    # dataset item 显式传 id 同理可以重复 seed 而不产生重复条目。
    langfuse.create_dataset(
        name=DATASET_NAME,
        description=(
            "Mini DeepResearch Agent 评测集：准确性 + 引用是否真的支持结论 + "
            "引用丰富度 + 报告质量（维度参考 DeepResearch Bench 的 RACE/FACT 框架）"
        ),
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
        evaluators=[
            accuracy_evaluator,
            citation_support_evaluator,
            citation_count_evaluator,
            report_quality_evaluator,
        ],
        # 每题要走完整的搜索+综合+反思流程，多题并发容易撞 Tavily/LLM 的速率限制，
        # 默认保守一点；调试单题时可以设更大的值。
        max_concurrency=int(os.getenv("EVAL_MAX_CONCURRENCY", "3")),
    )

    print(result.format(include_item_results=True))
    langfuse.flush()


if __name__ == "__main__":
    main()
