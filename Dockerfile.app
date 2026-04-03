# syntax=docker/dockerfile:1
# Aether 运行镜像：Rust gateway 直接服务 API + 前端静态文件
# 构建命令: docker build -f Dockerfile.app -t aether-app:latest .
# 用于 GitHub Actions CI（官方源）

# ==================== 前端构建 ====================
FROM node:22-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ==================== Rust gateway 构建 ====================
FROM rust:1.86-slim AS gateway-builder
WORKDIR /build
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    perl
COPY Cargo.toml Cargo.lock ./
COPY apps/ ./apps/
COPY crates/ ./crates/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    cargo build --release --locked -p aether-gateway && \
    cp target/release/aether-gateway /tmp/aether-gateway

# ==================== 运行时镜像 ====================
FROM debian:bookworm-slim

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libjemalloc2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    jemalloc_path="$(find /usr/lib -type f -name 'libjemalloc.so.2' | head -n1)"; \
    [ -n "$jemalloc_path" ]; \
    ln -sf "$jemalloc_path" /usr/local/lib/libjemalloc.so.2

# 复制 gateway 二进制
COPY --from=gateway-builder /tmp/aether-gateway /usr/local/bin/aether-gateway

# 复制前端构建产物
COPY --from=frontend-builder /app/frontend/dist /srv/frontend
RUN chmod -R 755 /srv/frontend

RUN mkdir -p /app/logs /app/data
WORKDIR /app

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 \
    MALLOC_CONF=background_thread:true,dirty_decay_ms:5000,muzzy_decay_ms:5000 \
    RUST_LOG=aether_gateway=info \
    AETHER_GATEWAY_BIND=0.0.0.0:80 \
    AETHER_GATEWAY_STATIC_DIR=/srv/frontend

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

ENTRYPOINT ["/usr/local/bin/aether-gateway"]
