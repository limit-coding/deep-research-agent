import os
import time
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import HTTPException, Request

PER_IP_MAX = int(os.getenv("DEMO_PER_IP_MAX", "3"))
PER_IP_WINDOW_SECONDS = int(os.getenv("DEMO_PER_IP_WINDOW_MINUTES", "10")) * 60
DAILY_MAX = int(os.getenv("DEMO_DAILY_MAX", "30"))

_ip_request_times: dict[str, list[float]] = defaultdict(list)
_daily_counts: dict[str, int] = defaultdict(int)


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def check_rate_limit(request: Request) -> None:
    """单 IP 滑动窗口 + 全局每日总额度，纯内存实现。

    这是公开 demo，省去 Redis：个人项目级别的流量，进程重启清零计数可以接受，
    换来的是不需要额外的基础设施依赖。
    """
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - PER_IP_WINDOW_SECONDS

    recent = [t for t in _ip_request_times[ip] if t > window_start]
    if len(recent) >= PER_IP_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"请求太频繁，每 {PER_IP_WINDOW_SECONDS // 60} 分钟最多 {PER_IP_MAX} 次，请稍后再试",
        )

    today = _today_key()
    if _daily_counts[today] >= DAILY_MAX:
        raise HTTPException(status_code=429, detail="今日 demo 额度已用完，请明天再来")

    recent.append(now)
    _ip_request_times[ip] = recent
    _daily_counts[today] += 1
