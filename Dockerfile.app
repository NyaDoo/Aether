# syntax=docker/dockerfile:1
# 运行镜像：从 base 提取产物到精简运行时
# 构建命令: docker build -f Dockerfile.app -t aether-app:latest .
# 用于 GitHub Actions CI（官方源）
FROM aether-base:latest AS builder
WORKDIR /app
# 复制前端源码并构建（CI 通过 no-cache-filters=builder 确保每次重建）
COPY frontend/ ./frontend/
RUN cd frontend && npm run build
# ==================== 运行时镜像 ====================
FROM python:3.14-slim
WORKDIR /app
# 运行时依赖（无 gcc/nodejs/npm，使用 BuildKit 缓存加速）
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    libpq5 \
    curl
# 从 base 镜像复制 Python 包
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
# 只复制需要的 Python 可执行文件
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/
COPY --from=builder /usr/local/bin/alembic /usr/local/bin/
# 从 builder 阶段复制前端构建产物
COPY --from=builder /app/frontend/dist /usr/share/nginx/html
RUN chmod -R 755 /usr/share/nginx/html
# 复制后端代码
COPY src/ ./src/
COPY alembic.ini ./
COPY alembic/ ./alembic/
COPY gunicorn_conf.py ./
# Nginx 配置模板
# 策略：白名单后端路由 → 后端代理，其余全部 → 前端 SPA（index.html）
# 智能处理 IP：有外层代理头就透传，没有就用直连 IP
RUN printf '%s\n' \
'map $http_x_real_ip $real_ip {' \
'    default $http_x_real_ip;' \
'    ""      $remote_addr;' \
'}' \
'' \
'map $http_x_forwarded_for $forwarded_for {' \
'    default $http_x_forwarded_for;' \
'    ""      $remote_addr;' \
'}' \
'' \
'server {' \
'    listen 80;' \
'    server_name _;' \
'    root /usr/share/nginx/html;' \
'    index index.html;' \
'    client_max_body_size 100M;' \
'' \
'    # gzip 压缩配置（对 base64 图片等非流式响应有效）' \
'    gzip on;' \
'    gzip_min_length 256;' \
'    gzip_comp_level 5;' \
'    gzip_vary on;' \
'    gzip_proxied any;' \
'    gzip_types application/json text/plain text/css text/javascript application/javascript application/octet-stream;' \
'    gzip_disable "msie6";' \
'' \
'    # 静态资源：长期缓存' \
'    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {' \
'        expires 1y;' \
'        add_header Cache-Control "public, no-transform";' \
'        try_files $uri =404;' \
'    }' \
'' \
'    # 安全：阻止访问源码目录' \
'    location ~ ^/(src|node_modules)/ {' \
'        deny all;' \
'        return 404;' \
'    }' \
'' \
'    # 后端 API 路由（白名单）→ 代理到后端' \
'    location ~ ^/(api|v1|health)(/|$) {' \
'        proxy_pass http://127.0.0.1:PORT_PLACEHOLDER;' \
'        proxy_http_version 1.1;' \
'        proxy_set_header Host $host;' \
'        proxy_set_header X-Real-IP $real_ip;' \
'        proxy_set_header X-Forwarded-For $forwarded_for;' \
'        proxy_set_header X-Forwarded-Proto $scheme;' \
'        proxy_set_header Connection "";' \
'        proxy_set_header Accept $http_accept;' \
'        proxy_set_header Content-Type $content_type;' \
'        proxy_set_header Authorization $http_authorization;' \
'        proxy_set_header X-Api-Key $http_x_api_key;' \
'        proxy_buffering off;' \
'        proxy_cache off;' \
'        proxy_request_buffering off;' \
'        chunked_transfer_encoding on;' \
'        gzip off;' \
'        add_header X-Accel-Buffering no;' \
'        proxy_connect_timeout 600s;' \
'        proxy_send_timeout 600s;' \
'        proxy_read_timeout 600s;' \
'    }' \
'' \
'    # API 文档路由 → 代理到后端' \
'    location ~ ^/(docs|redoc|openapi\\.json)$ {' \
'        proxy_pass http://127.0.0.1:PORT_PLACEHOLDER;' \
'        proxy_http_version 1.1;' \
'        proxy_set_header Host $host;' \
'        proxy_set_header X-Real-IP $real_ip;' \
'        proxy_set_header X-Forwarded-For $forwarded_for;' \
'        proxy_set_header X-Forwarded-Proto $scheme;' \
'    }' \
'' \
'    # 所有其他路由 → 前端 SPA（先尝试静态文件，再回退到 index.html）' \
'    location / {' \
'        try_files $uri $uri/ /index.html;' \
'    }' \
'}' > /etc/nginx/sites-available/default.template
# Supervisor 配置
RUN printf '%s\n' \
'[supervisord]' \
'nodaemon=true' \
'logfile=/var/log/supervisor/supervisord.log' \
'pidfile=/var/run/supervisord.pid' \
'' \
'[program:nginx]' \
'command=/bin/bash -c "sed \"s/PORT_PLACEHOLDER/8084/g\" /etc/nginx/sites-available/default.template > /etc/nginx/sites-available/default && /usr/sbin/nginx -g \"daemon off;\""' \
'autostart=true' \
'autorestart=true' \
'stdout_logfile=/var/log/nginx/access.log' \
'stderr_logfile=/var/log/nginx/error.log' \
'' \
'[program:app]' \
'command=gunicorn src.main:app -c gunicorn_conf.py --preload -w %(ENV_GUNICORN_WORKERS)s -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8084 --timeout 120 --max-requests 2000 --max-requests-jitter 100 --access-logfile - --error-logfile - --log-level info' \
'directory=/app' \
'autostart=true' \
'autorestart=true' \
'stdout_logfile=/dev/stdout' \
'stdout_logfile_maxbytes=0' \
'stderr_logfile=/dev/stderr' \
'stderr_logfile_maxbytes=0' \
'environment=PYTHONUNBUFFERED=1,PYTHONIOENCODING=utf-8,LANG=C.UTF-8,LC_ALL=C.UTF-8,DOCKER_CONTAINER=true' > /etc/supervisor/conf.d/supervisord.conf
# 创建目录
RUN mkdir -p /var/log/supervisor /app/logs /app/data
# 入口脚本（启动前执行迁移）
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
# 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PORT=8084 \
    GUNICORN_WORKERS=4
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
