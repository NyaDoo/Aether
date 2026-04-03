"""
图片 URL 解析器

当跨格式转换时，目标格式需要 base64 图片数据（如 Claude 不原生支持 URL 图片引用），
该模块负责自动下载图片 URL 并转换为 base64 内嵌数据。

使用方式：在跨格式转换前/后调用 resolve_image_urls()。
"""

import asyncio
import base64
import ipaddress
import mimetypes
import socket
from urllib.parse import urljoin, urlparse

import httpx

from src.core.api_format.conversion.internal import (
    FileBlock,
    ImageBlock,
    InternalRequest,
)
from src.core.logger import logger

# 需要 base64 图片数据的目标格式前缀
_FORMATS_REQUIRING_BASE64 = frozenset({"CLAUDE"})

# 图片下载超时（秒）
_DOWNLOAD_TIMEOUT = 15.0

# 单张图片最大大小（字节，20MB）
_MAX_IMAGE_SIZE = 20 * 1024 * 1024

# 并发下载数量限制
_MAX_CONCURRENT_DOWNLOADS = 8

# 最大重定向次数
_MAX_REDIRECTS = 5


def _is_private_ip(addr: str) -> bool:
    """检查 IP 地址是否为私有/内网地址。"""
    try:
        ip = ipaddress.ip_address(addr)
        return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)
    except ValueError:
        return True


async def _resolve_and_validate_host(hostname: str) -> list[str] | None:
    """DNS 解析并校验所有 IP 均为公网地址（SSRF 防护）。

    返回已校验的公网 IP 列表；如果任一 IP 为私有地址或解析失败，返回 None。
    """
    try:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        ips: list[str] = []
        for _family, _type, _proto, _canonname, sockaddr in infos:
            addr = sockaddr[0]
            if _is_private_ip(addr):
                return None
            ips.append(addr)
        return ips or None
    except (socket.gaierror, ValueError, OSError):
        # DNS 解析失败：安全默认拒绝
        return None


async def _validate_url(url: str) -> bool:
    """校验 URL 的 scheme 和主机地址，通过返回 True，否则返回 False。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        logger.warning("[ImageResolver] 不支持的 URL scheme: {}", url[:100])
        return False
    hostname = parsed.hostname or ""
    resolved = await _resolve_and_validate_host(hostname)
    if resolved is None:
        logger.warning("[ImageResolver] 拒绝下载私有网络地址: {}", url[:100])
        return False
    return True


def _validate_peer_ip(resp: httpx.Response) -> bool:
    """校验 HTTP 响应的实际对端 IP 是否为公网地址（防 DNS rebinding TOCTOU 绕过）。

    httpx 通过 extensions["network_stream"] 暴露底层连接，
    从中可获取对端地址进行二次校验。
    """
    try:
        network_stream = resp.extensions.get("network_stream")
        if network_stream is None:
            logger.debug("[ImageResolver] network_stream 不可用, DNS rebinding 检测跳过")
            return True
        # asyncio transport 标准 extra info key 是 peername
        peername = network_stream.get_extra_info("peername")
        if peername is not None:
            peer_ip = peername[0] if isinstance(peername, tuple) else str(peername)
            if _is_private_ip(peer_ip):
                logger.warning("[ImageResolver] DNS rebinding 检测: 实际连接到私有 IP {}", peer_ip)
                return False
    except Exception as e:
        # 无法获取对端信息时放行（不阻塞正常功能），依赖前置 DNS 校验
        logger.debug("[ImageResolver] 无法获取对端 IP 信息（DNS rebinding 检测跳过）: {}", e)
    return True


async def _download_file(
    client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore
) -> tuple[str, str] | None:
    """下载 URL 并返回 (base64_data, media_type)，失败返回 None。

    手动处理重定向，每一跳都检查目标地址（防止重定向到内网的 SSRF 绕过）。
    连接建立后二次校验对端 IP（防 DNS rebinding）。
    """
    async with semaphore:
        try:
            if not await _validate_url(url):
                return None

            current_url = url
            for _ in range(_MAX_REDIRECTS + 1):
                async with client.stream("GET", current_url) as resp:
                    # DNS rebinding 防护：校验实际连接的对端 IP
                    if not _validate_peer_ip(resp):
                        return None

                    if resp.is_redirect:
                        location = resp.headers.get("location", "")
                        if not location:
                            logger.warning("[ImageResolver] 重定向缺少 Location: {}", url[:100])
                            return None
                        redirect_url = urljoin(str(current_url), location)
                        if not await _validate_url(redirect_url):
                            return None
                        current_url = redirect_url
                        continue

                    resp.raise_for_status()

                    content_type = resp.headers.get("content-type", "")
                    media_type = content_type.split(";")[0].strip().lower()
                    if not media_type:
                        media_type = _guess_media_type(current_url)

                    # 检查 MIME 类型是否为目标 API 可接受的类型
                    if not any(media_type.startswith(p) for p in _ACCEPTED_MIME_PREFIXES):
                        logger.warning(
                            "[ImageResolver] 不支持的 MIME 类型 {}, 跳过: {}",
                            media_type,
                            url[:100],
                        )
                        return None

                    # 预检 Content-Length（如果有）
                    content_length = resp.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > _MAX_IMAGE_SIZE:
                                logger.warning(
                                    "[ImageResolver] Content-Length 超过大小限制: {}",
                                    url[:100],
                                )
                                return None
                        except (TypeError, ValueError):
                            pass

                    # 流式累计读取并检查大小
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in resp.aiter_bytes():
                        total += len(chunk)
                        if total > _MAX_IMAGE_SIZE:
                            logger.warning(
                                "[ImageResolver] 文件超过大小限制 ({} bytes > {}): {}",
                                total,
                                _MAX_IMAGE_SIZE,
                                url[:100],
                            )
                            return None
                        chunks.append(chunk)

                    data = b"".join(chunks)
                    b64 = base64.b64encode(data).decode("ascii")
                    return b64, media_type

            logger.warning("[ImageResolver] 超过最大重定向次数: {}", url[:100])
            return None
        except Exception as e:
            logger.warning("[ImageResolver] 下载文件失败: {} - {}", url[:100], e)
            return None


def _guess_media_type(url: str) -> str:
    """从 URL 路径猜测 MIME 类型。"""
    path = url.split("?")[0]
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"


# Claude API 支持的 MIME 类型前缀（图片/文档/音频等可直接作为 base64 内嵌的类型）
_ACCEPTED_MIME_PREFIXES: tuple[str, ...] = (
    "image/",
    "application/pdf",
    "text/",
    "audio/",
    "video/",
)


def _is_data_url(url: str) -> bool:
    """判断是否是 data: URL（已内嵌 base64）。"""
    return url.startswith("data:")


async def resolve_image_urls(
    internal: InternalRequest,
    target_format: str,
) -> None:
    """遍历 InternalRequest 中所有 ImageBlock/FileBlock，对有 url 无 data 的进行下载转 base64。

    仅当 target_format 需要 base64 时执行下载（如 CLAUDE）。
    直接修改 internal 对象，无返回值。
    """
    target_upper = str(target_format).upper()

    # 检查目标格式是否需要 base64
    needs_base64 = any(target_upper.startswith(prefix) for prefix in _FORMATS_REQUIRING_BASE64)
    if not needs_base64:
        return

    # 收集所有需要下载的 block 及其 URL
    download_items: list[tuple[ImageBlock | FileBlock, str]] = []
    for msg in internal.messages:
        for block in msg.content:
            if isinstance(block, ImageBlock):
                if block.url and not block.data and not _is_data_url(block.url):
                    download_items.append((block, block.url))
            elif isinstance(block, FileBlock):
                if block.file_url and not block.data and not _is_data_url(block.file_url):
                    download_items.append((block, block.file_url))

    if not download_items:
        return

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_DOWNLOADS)

    # 复用同一个 client 并发下载所有文件（禁用自动重定向，在 _download_file 中手动处理）
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(_DOWNLOAD_TIMEOUT),
        limits=httpx.Limits(max_connections=_MAX_CONCURRENT_DOWNLOADS),
    ) as client:
        tasks = [_download_file(client, url, semaphore) for _, url in download_items]
        results = await asyncio.gather(*tasks)

    # 回写结果
    for (block, url), result in zip(download_items, results):
        if result is not None:
            b64_data, media_type = result
            block.data = b64_data
            if not block.media_type:
                block.media_type = media_type
            # 保留原始 URL 以便调试，清除源 URL 字段避免语义模糊
            block.extra["original_url"] = url
            if isinstance(block, ImageBlock):
                block.url = None
            elif isinstance(block, FileBlock):
                block.file_url = None


__all__ = ["resolve_image_urls"]
