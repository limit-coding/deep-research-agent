import logging

from langfuse import observe
from tavily import TavilyClient
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient()  # 从 TAVILY_API_KEY 环境变量读取
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _search_with_retry(client: TavilyClient, query: str, max_results: int, search_depth: str) -> dict:
    """直接调 Tavily API，失败抛异常——由外层 retry 决定重试。

    单独抽成一个函数是为了让 @retry 装饰器只装饰实际的网络调用，
    而不是整个 web_search（web_search 里还有异常处理和结果格式化逻辑，
    那些不需要重试）。
    """
    return client.search(query, max_results=max_results, search_depth=search_depth)


@observe(as_type="tool", name="web_search")
def web_search(query: str, max_results: int = 4, search_depth: str = "basic") -> list[dict]:
    """统一的搜索接口，返回结构化的 {query, title, url, snippet} 列表。

    自己包一层而不是直接在各处调用 TavilyClient：以后想换 Tavily 以外的搜索源
    （比如免 key 的 DuckDuckGo），只需要改这一个函数的实现，nodes.py 不用动。

    瞬时网络错误/限流（429）会触发最多 3 次重试，指数退避 1→2→4→8 秒。
    3 次全部失败后返回空列表——交给 reflect 节点去判断"信息不足需要重试"，
    而不是让整条图因为一次搜索失败就崩掉。

    search_depth="advanced" 比 "basic" 抽取更完整的页面内容，但 Tavily 按更高 credit 计费、
    延迟也更高——是否值得这个代价由调用方（search_node）按"这轮是不是已确认信息不足"来决定，
    这个函数本身不做判断。
    """
    try:
        response = _search_with_retry(_get_client(), query, max_results, search_depth)
    except Exception:
        logger.warning("web_search failed after retries for query=%r", query, exc_info=True)
        return []

    return [
        {
            "query": query,
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]
