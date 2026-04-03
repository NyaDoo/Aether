"""
格式转换内部表示（Internal / Canonical Format）

该模块定义 Hub-and-Spoke 架构的"中间表示法"，用于把不同 Provider 的请求/响应/流式事件
统一映射到稳定的内部结构，再转换为目标格式。

设计原则：
- 类型安全：尽量用 dataclass + Enum 表达语义，便于 IDE/静态检查
- 可扩展：未知/不可逆字段写入 extra/raw，避免静默丢失
- 兼容优先：UnknownBlock 在内部保留，但默认在输出阶段丢弃（可观测、可随时调整策略）

字段修改须知：
- 本文件是所有 normalizer 的共享契约，修改字段语义会同时影响所有格式的输入输出
- 每个字段的注释标注了各格式的映射关系（OpenAI/Claude/Gemini）
- 修改前请检查 tests/core/api_format/conversion/ 下的 roundtrip + schema 测试
- 新增字段应标注 "可选" 并给默认值，避免破坏现有 normalizer
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"
    TOOL = "tool"
    UNKNOWN = "unknown"


class ContentType(str, Enum):
    TEXT = "text"
    THINKING = "thinking"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    UNKNOWN = "unknown"


class StopReason(str, Enum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    # Claude streaming 里会出现（官方文档枚举）：pause_turn / refusal
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"
    CONTENT_FILTERED = "content_filtered"
    UNKNOWN = "unknown"


class ErrorType(str, Enum):
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    SERVER_ERROR = "server_error"
    CONTENT_FILTERED = "content_filtered"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    UNKNOWN = "unknown"


@dataclass
class TextBlock:
    """文本内容块

    Format mapping:
      OpenAI: message.content (string) / content[].type="text"
      Claude: content[].type="text"
      Gemini: parts[].text
    """

    type: ContentType = field(default=ContentType.TEXT, init=False)
    text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThinkingBlock:
    """思考过程内容块

    Format mapping:
      OpenAI: message.reasoning_content / delta.reasoning_content
      Claude: content[].type="thinking" (thinking + signature)
      Gemini: parts[].thought=true (text + thoughtSignature)
    """

    type: ContentType = field(default=ContentType.THINKING, init=False)
    thinking: str = ""
    # Claude signature / Gemini thoughtSignature; OpenAI 无对应字段
    signature: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageBlock:
    """图片内容块

    Format mapping:
      OpenAI: content[].type="image_url" -> image_url.url (URL or data:mime;base64,...)
      Claude: content[].type="image" -> source.type="base64" | source.type="url"
      Gemini: parts[].inlineData (base64) / parts[].fileData (URI)
    """

    type: ContentType = field(default=ContentType.IMAGE, init=False)
    data: str | None = None  # base64 encoded image data (mutually exclusive with url)
    media_type: str | None = None  # MIME type, e.g. "image/png"
    url: str | None = None  # URL reference (mutually exclusive with data)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolUseBlock:
    """工具调用内容块

    Format mapping:
      OpenAI: message.tool_calls[].id / .function.name / .function.arguments(JSON str)
      Claude: content[].type="tool_use" -> id / name / input(dict)
      Gemini: parts[].functionCall -> name / args(dict); id 由 normalizer 生成

    Contract:
      tool_id:    roundtrip 保留; OpenAI/Claude 原生提供, Gemini 由 normalizer 合成
      tool_name:  必须非空
      tool_input: 已解析的 dict (非 JSON 字符串)
    """

    type: ContentType = field(default=ContentType.TOOL_USE, init=False)
    tool_id: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultBlock:
    """工具结果内容块

    Format mapping:
      OpenAI: role="tool" message -> tool_call_id + content(string)
      Claude: content[].type="tool_result" -> tool_use_id + content
      Gemini: parts[].functionResponse -> name + response(dict)

    Contract:
      tool_use_id:  关联 ToolUseBlock.tool_id; OpenAI/Claude 必须非空
      tool_name:    Gemini functionResponse.name 需要; OpenAI/Claude 可为 None
      output:       结构化输出 (dict/list); 与 content_text 二选一
      content_text: 纯文本输出; 与 output 二选一
    """

    type: ContentType = field(default=ContentType.TOOL_RESULT, init=False)
    tool_use_id: str = ""
    tool_name: str | None = None
    output: Any = None
    content_text: str | None = None
    is_error: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileBlock:
    """文件内容块（PDF、文档等）

    Format mapping:
      OpenAI: content[].type="file" -> file.file_data(data URL) / file.file_id
      Claude: content[].type="document" -> source.type="base64" / source.type="url"
      Gemini: parts[].fileData -> fileUri + mimeType
    """

    type: ContentType = field(default=ContentType.FILE, init=False)
    data: str | None = None  # base64 encoded file data
    media_type: str | None = None  # MIME type
    file_id: str | None = None  # OpenAI file reference
    file_url: str | None = None  # Gemini fileData URI
    filename: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioBlock:
    """音频内容块

    Format mapping:
      OpenAI: content[].type="input_audio" -> input_audio.data + input_audio.format
      Claude: content[].type="audio" (planned)
      Gemini: parts[].inlineData (audio MIME)
    """

    type: ContentType = field(default=ContentType.AUDIO, init=False)
    data: str | None = None  # base64 encoded audio data
    media_type: str | None = None  # full MIME (e.g. audio/mp3)
    format: str | None = None  # short format name (e.g. mp3, wav)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnknownBlock:
    """未知内容块（用于前向兼容）"""

    type: ContentType = field(default=ContentType.UNKNOWN, init=False)
    raw_type: str = ""  # 原始的类型字符串（各格式不一致）
    payload: dict[str, Any] = field(default_factory=dict)  # 原始结构（尽量保持）
    extra: dict[str, Any] = field(default_factory=dict)


ContentBlock = (
    TextBlock
    | ThinkingBlock
    | ImageBlock
    | FileBlock
    | AudioBlock
    | ToolUseBlock
    | ToolResultBlock
    | UnknownBlock
)


@dataclass
class InternalMessage:
    """统一的消息表示"""

    role: Role
    content: list[ContentBlock]  # 统一使用列表，纯文本用单个 TextBlock
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """统一的工具定义"""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None  # JSON Schema
    extra: dict[str, Any] = field(default_factory=dict)


class ToolChoiceType(str, Enum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    TOOL = "tool"


@dataclass
class ToolChoice:
    """统一的工具选择"""

    type: ToolChoiceType
    tool_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class InstructionSegment:
    """系统/开发者指令段

    OpenAI 区分 system/developer 两种 role, Claude/Gemini 只有 system string.
    instructions 列表保留 OpenAI 的 role 语义和顺序, system 字段是 join 后的纯文本兜底.
    """

    role: Role  # Role.SYSTEM / Role.DEVELOPER only
    text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThinkingConfig:
    """统一的思考/推理配置

    Format mapping:
      OpenAI: reasoning_effort ("low"/"medium"/"high") -> budget_tokens via lookup table
      Claude: thinking.type="enabled" + thinking.budget_tokens
      Gemini: generationConfig.thinkingConfig.thinkingBudget
    """

    enabled: bool = False
    budget_tokens: int | None = None  # None = provider default
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseFormatConfig:
    """统一的响应格式配置（JSON mode / structured output）"""

    type: str = "text"  # "text" | "json_object" | "json_schema"
    json_schema: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalRequest:
    """统一的请求表示

    Format mapping (key fields):
      model:        OpenAI/Claude body.model; Gemini URL path param
      instructions: OpenAI system/developer messages; Claude/Gemini -> join to system string
      system:       instructions join fallback; Claude system param; Gemini systemInstruction
      max_tokens:   OpenAI max_tokens/max_completion_tokens; Claude max_tokens; Gemini maxOutputTokens
      tools:        OpenAI tools[].function; Claude tools[]; Gemini tools[].functionDeclarations
      tool_choice:  OpenAI tool_choice; Claude tool_choice; Gemini toolConfig.functionCallingConfig
    """

    model: str
    messages: list[InternalMessage]

    # 指令层：保留 system/developer 结构与顺序
    instructions: list[InstructionSegment] = field(default_factory=list)

    # 兼容字段：instructions 的 join 文本（无 role 标签），用于 Claude/Gemini 这类仅接受字符串 system 的格式
    system: str | None = None

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None  # auto/none/required 或指定 tool_name

    # 思考/推理配置
    thinking: ThinkingConfig | None = None

    # 并行工具调用控制
    parallel_tool_calls: bool | None = None

    # 采样参数
    n: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None

    # 响应格式
    response_format: ResponseFormatConfig | None = None

    # 模型输出上限（来自 GlobalModel.config.output_limit，用于跨格式转换时的 max_tokens 默认值）
    output_limit: int | None = None

    extra: dict[str, Any] = field(default_factory=dict)  # 未识别字段透传

    def to_debug_dict(self) -> dict[str, Any]:
        """用于日志和调试的简化表示"""
        return {
            "model": self.model,
            "instruction_count": len(self.instructions),
            "message_count": len(self.messages),
            "has_system": bool(self.instructions) or bool(self.system),
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "tool_count": len(self.tools) if self.tools else 0,
            "extra_keys": list(self.extra.keys()),
        }


@dataclass
class UsageInfo:
    """统一的使用量信息"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalResponse:
    """统一的响应表示

    Format mapping:
      id:          OpenAI id; Claude id; Gemini (none, synthesized)
      model:       OpenAI model; Claude model; Gemini model (from metadata)
      content:     OpenAI choices[0].message; Claude content[]; Gemini candidates[0].content.parts
      stop_reason: OpenAI finish_reason; Claude stop_reason; Gemini finishReason
      usage:       OpenAI usage; Claude usage; Gemini usageMetadata
    """

    id: str
    model: str
    content: list[ContentBlock]
    stop_reason: StopReason | None = None
    usage: UsageInfo | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_debug_dict(self) -> dict[str, Any]:
        """用于日志和调试的简化表示"""
        usage = None
        if self.usage:
            usage = {
                "input": self.usage.input_tokens,
                "output": self.usage.output_tokens,
            }
        return {
            "id": self.id,
            "model": self.model,
            "content_block_count": len(self.content),
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "usage": usage,
            "extra_keys": list(self.extra.keys()),
        }


@dataclass
class InternalError:
    """统一的错误表示"""

    type: ErrorType
    message: str
    code: str | None = None  # 原始错误码
    param: str | None = None  # 导致错误的参数
    retryable: bool = False  # 是否可重试
    extra: dict[str, Any] = field(default_factory=dict)

    def to_debug_dict(self) -> dict[str, Any]:
        """用于日志和调试"""
        return {
            "type": self.type.value,
            "message": self.message,
            "code": self.code,
            "param": self.param,
            "retryable": self.retryable,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class FormatCapabilities:
    supports_stream: bool = True
    supports_error_conversion: bool = True
    supports_tools: bool = True
    supports_images: bool = False
    supported_features: frozenset[str] = field(default_factory=frozenset)


__all__ = [
    "Role",
    "ContentType",
    "StopReason",
    "ErrorType",
    "ToolChoiceType",
    "TextBlock",
    "ThinkingBlock",
    "ImageBlock",
    "FileBlock",
    "AudioBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "UnknownBlock",
    "ContentBlock",
    "InternalMessage",
    "InstructionSegment",
    "ToolDefinition",
    "ToolChoice",
    "ThinkingConfig",
    "ResponseFormatConfig",
    "InternalRequest",
    "UsageInfo",
    "InternalResponse",
    "InternalError",
    "FormatCapabilities",
]
