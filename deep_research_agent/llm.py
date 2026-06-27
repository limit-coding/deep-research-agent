import os


def get_llm():
    """按 LLM_PROVIDER 环境变量返回对应的 chat model。

    迭代阶段要在 Haiku / GPT-4o-mini 之间快速比较成本和质量，
    把 provider 选择收在这一处，nodes.py 不需要关心具体用哪家。
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    # max_retries=3：交给各家 SDK 的内置重试（指数退避）处理 429/5xx。
    # Tavily 那边用 tenacity 显式控制是因为我们自己包了 TavilyClient，没有 LangChain 托管；
    # LLM 这边走 LangChain，直接用 SDK 参数，避免双层重试逻辑互相干扰。
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        return ChatAnthropic(model=model, temperature=0, max_retries=3)  # 固定 0：评测时同一题多次跑要可比
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0, max_retries=3)
    elif provider == "deepseek":
        # DeepSeek 的 API 跟 OpenAI 兼容，复用 ChatOpenAI 换个 base_url/key 即可，
        # 不用单独装 langchain-deepseek 这类额外依赖。
        from langchain_openai import ChatOpenAI

        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        return ChatOpenAI(
            model=model,
            temperature=0,
            max_retries=3,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    raise ValueError(f"未知的 LLM_PROVIDER: {provider}")


def get_structured_llm(schema):
    """统一走 tool-calling 的 structured output，而不是各家 SDK 默认值。

    ChatOpenAI 的 with_structured_output 默认 method="json_schema"（OpenAI 的严格模式），
    但 DeepSeek 等 OpenAI 兼容服务不支持这个 response_format，会直接 400。
    function_calling 是 Anthropic/OpenAI/DeepSeek 三家都支持的最大公约数，
    所以在这一处统一指定，调用方（nodes.py、eval）不需要关心具体用哪家。
    """
    return get_llm().with_structured_output(schema, method="function_calling")
