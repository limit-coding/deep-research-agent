import logging

from tavily import TavilyClient

logger = logging.getLogger(__name__)

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient()  # 从 TAVILY_API_KEY 环境变量读取
    return _client

"""就一个函数 web_search(query)，里面调
  Tavily，失败了就返回空列表而不是报错崩掉——交给 reflect
  节点去发现"这块信息没查到，要不要重试"，而不是让一次网络抖动炸掉整条流程。"""
def web_search(query: str, max_results: int = 4) -> list[dict]:
    """统一的搜索接口，返回结构化的 {query, title, url, snippet} 列表。

    自己包一层而不是直接在各处调用 TavilyClient：以后想换 Tavily 以外的搜索源
    （比如免 key 的 DuckDuckGo），只需要改这一个函数的实现，nodes.py 不用动。
    网络/限流失败时返回空列表而不是抛异常——交给 reflect 节点去判断"信息不足需要重试"，
    而不是让整条图因为一次搜索失败就崩掉。
    """
    try:
        response = _get_client().search(query, max_results=max_results)
    except Exception:
        logger.warning("web_search failed for query=%r", query, exc_info=True)
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
