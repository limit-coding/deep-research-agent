import os


def get_llm():
    """按 LLM_PROVIDER 环境变量返回对应的 chat model。

    迭代阶段要在 Haiku / GPT-4o-mini 之间快速比较成本和质量，
    把 provider 选择收在这一处，nodes.py 不需要关心具体用哪家。
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        return ChatAnthropic(model=model, temperature=0)  # 固定 0：评测时同一题多次跑要可比
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)

    raise ValueError(f"未知的 LLM_PROVIDER: {provider}")
