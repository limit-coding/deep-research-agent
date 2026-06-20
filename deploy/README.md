# 部署说明

这两个文件是版本控制的参考模板，不是直接可执行的脚本——服务器上的真实文件路径、域名是定死的，
存进 repo 是为了部署思路可追溯（面试时也方便直接照着讲），不是为了多机器复用。

## 部署到一台新机器需要的步骤

1. `git clone` 这个 repo 到目标目录（约定跟同机器上其他项目一致：`/home/<user>/deep-research-agent`）
2. `uv sync --extra api`
3. `.env` 填真实 key（`ANTHROPIC_API_KEY`/`DEEPSEEK_API_KEY` 等 + `TAVILY_API_KEY` + `LANGFUSE_*`），
   再加上 demo 专属的几个：
   - `ALLOWED_ORIGINS=https://research.learnpath.tech`（CORS，多个用逗号分隔）
   - `DEMO_PER_IP_MAX` / `DEMO_PER_IP_WINDOW_MINUTES` / `DEMO_DAILY_MAX`（限流，留空则用 `api/rate_limit.py` 里的默认值）
4. `cd frontend && npm install && npm run build`（产出 `frontend/dist`，nginx 直接当静态资源托管）
5. 把 `deepresearch-backend.service` 拷到 `/etc/systemd/system/`，按实际路径/端口调整后：
   `systemctl daemon-reload && systemctl enable --now deepresearch-backend`
6. 把 `research.learnpath.tech.nginx.conf` 拷到 `/etc/nginx/sites-available/research.learnpath.tech`，
   `ln -s` 到 `sites-enabled`，`nginx -t` 通过后 `systemctl reload nginx`
7. DNS：给域名加一条 A 记录指向服务器公网 IP（这一步只能在域名注册商/DNS 控制台手动加）
8. DNS 生效后：`certbot --nginx -d research.learnpath.tech` 签证书（会自动改写上面 nginx 配置里的证书路径）

## 端口约定

后端只 bind `127.0.0.1:18002`，不开公网端口——所有外部流量都经过 nginx 的 80/443，
跟这台机器上其他服务一致的约定，不需要额外开 ufw 规则。
