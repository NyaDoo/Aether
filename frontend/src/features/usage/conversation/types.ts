/**
 * 对话解析器类型定义
 * 统一各种 API 格式（Claude/OpenAI/Gemini）的消息解析接口
 */

// ============================================================
// API 格式枚举
// ============================================================

export type ApiFormat = 'claude' | 'openai' | 'gemini' | 'unknown'

// ============================================================
// 消息类型定义
// ============================================================

/** 消息角色 */
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool'

/** 内容块类型 */
export type ContentBlockType =
  | 'text'        // 普通文本
  | 'thinking'    // 思考过程（Claude extended thinking）
  | 'tool_use'    // 工具调用
  | 'tool_result' // 工具结果
  | 'image'       // 图片
  | 'file'        // 文件
  | 'code'        // 代码块
  | 'error'       // 错误信息

/** 内容块基础接口 */
export interface ContentBlockBase {
  type: ContentBlockType
}

/** 文本内容块 */
export interface TextContentBlock extends ContentBlockBase {
  type: 'text'
  text: string
}

/** 思考内容块 */
export interface ThinkingContentBlock extends ContentBlockBase {
  type: 'thinking'
  thinking: string
  /** 签名（用于验证，可选） */
  signature?: string
}

/** 工具调用内容块 */
export interface ToolUseContentBlock extends ContentBlockBase {
  type: 'tool_use'
  toolId: string
  toolName: string
  input: Record<string, any> | string
}

/** 工具结果内容块 */
export interface ToolResultContentBlock extends ContentBlockBase {
  type: 'tool_result'
  toolUseId: string
  content: string | ContentBlock[]
  isError?: boolean
}

/** 图片内容块 */
export interface ImageContentBlock extends ContentBlockBase {
  type: 'image'
  /** 图片来源类型 */
  sourceType: 'base64' | 'url'
  /** Base64 数据或 URL */
  data?: string
  url?: string
  /** MIME 类型 */
  mimeType?: string
  /** 图片描述（alt text） */
  alt?: string
}

/** 文件内容块 */
export interface FileContentBlock extends ContentBlockBase {
  type: 'file'
  fileName: string
  fileType?: string
  fileSize?: number
  content?: string
}

/** 代码内容块 */
export interface CodeContentBlock extends ContentBlockBase {
  type: 'code'
  language?: string
  code: string
}

/** 错误内容块 */
export interface ErrorContentBlock extends ContentBlockBase {
  type: 'error'
  message: string
  code?: string
}

/** 所有内容块类型的联合 */
export type ContentBlock =
  | TextContentBlock
  | ThinkingContentBlock
  | ToolUseContentBlock
  | ToolResultContentBlock
  | ImageContentBlock
  | FileContentBlock
  | CodeContentBlock
  | ErrorContentBlock

// ============================================================
// 消息定义
// ============================================================

/** 解析后的消息 */
export interface ParsedMessage {
  /** 消息角色 */
  role: MessageRole
  /** 内容块列表 */
  content: ContentBlock[]
}

/** 解析后的对话 */
export interface ParsedConversation {
  /** 系统提示词 */
  system?: string
  /** 消息列表 */
  messages: ParsedMessage[]
  /** 是否为流式响应 */
  isStream: boolean
  /** 解析错误（如果有） */
  parseError?: string
  /** 原始 API 格式 */
  apiFormat: ApiFormat
  /** 模型名称 */
  model?: string
}

// ============================================================
// 解析器接口
// ============================================================

/** API 格式检测器 */
export interface FormatDetector {
  /**
   * 检测是否匹配该格式
   * @param requestBody 请求体
   * @param responseBody 响应体
   * @param hint 后端提供的格式提示
   * @returns 匹配置信度 (0-100)，0 表示不匹配
   */
  detect(requestBody: any, responseBody: any, hint?: string): number
}

/** 请求体解析器 */
export interface RequestParser {
  /**
   * 解析请求体
   * @param requestBody 请求体
   * @returns 解析后的对话
   */
  parseRequest(requestBody: any): ParsedConversation
}

/** 响应体解析器 */
export interface ResponseParser {
  /**
   * 解析响应体
   * @param responseBody 响应体
   * @returns 解析后的对话
   */
  parseResponse(responseBody: any): ParsedConversation

  /**
   * 解析流式响应
   * @param chunks 响应块列表
   * @returns 解析后的对话
   */
  parseStreamResponse(chunks: any[]): ParsedConversation
}

/** 完整的 API 格式解析器 */
export interface ApiFormatParser extends FormatDetector, RequestParser, ResponseParser {
  /** 格式名称 */
  readonly format: ApiFormat
  /** 格式显示名称 */
  readonly displayName: string

  /**
   * 渲染请求体为渲染块
   * @param requestBody 请求体
   * @returns 渲染结果
   */
  renderRequest(requestBody: any): import('./render').RenderResult

  /**
   * 渲染响应体为渲染块
   * @param responseBody 响应体
   * @returns 渲染结果
   */
  renderResponse(responseBody: any): import('./render').RenderResult
}

// ============================================================
// 流式响应相关
// ============================================================

/** 流式响应元数据 */
export interface StreamMetadata {
  stream: boolean
}

/** 流式响应体结构 */
export interface StreamResponseBody {
  metadata?: StreamMetadata
  chunks?: any[]
}

/** 检查是否为流式响应 */
export function isStreamResponse(body: any): body is StreamResponseBody {
  return body?.metadata?.stream === true && Array.isArray(body?.chunks)
}

// ============================================================
// 工具函数
// ============================================================

/** 创建文本内容块 */
export function createTextBlock(text: string): TextContentBlock {
  return { type: 'text', text }
}

/** 创建思考内容块 */
export function createThinkingBlock(thinking: string, signature?: string): ThinkingContentBlock {
  return { type: 'thinking', thinking, signature }
}

/** 创建工具调用内容块 */
export function createToolUseBlock(
  toolId: string,
  toolName: string,
  input: Record<string, any> | string
): ToolUseContentBlock {
  return { type: 'tool_use', toolId, toolName, input }
}

/** 创建工具结果内容块 */
export function createToolResultBlock(
  toolUseId: string,
  content: string | ContentBlock[],
  isError?: boolean
): ToolResultContentBlock {
  return { type: 'tool_result', toolUseId, content, isError }
}

/** 创建图片内容块 */
export function createImageBlock(
  sourceType: 'base64' | 'url',
  options: { data?: string; url?: string; mimeType?: string; alt?: string }
): ImageContentBlock {
  return { type: 'image', sourceType, ...options }
}

/** 创建错误内容块 */
export function createErrorBlock(message: string, code?: string): ErrorContentBlock {
  return { type: 'error', message, code }
}

/** 创建空的解析结果 */
export function createEmptyConversation(
  apiFormat: ApiFormat = 'unknown',
  parseError?: string
): ParsedConversation {
  return {
    messages: [],
    isStream: false,
    apiFormat,
    parseError,
  }
}

/** 创建解析后的消息 */
export function createMessage(role: MessageRole, content: ContentBlock[]): ParsedMessage {
  return { role, content }
}

// ============================================================
// 对话轮次相关类型
// ============================================================

/** 轮次统计信息 */
export interface TurnStats {
  /** 用户消息字符数 */
  userChars: number
  /** 助手消息字符数 */
  assistantChars: number
  /** 是否包含思考过程 */
  hasThinking: boolean
  /** 是否包含工具调用 */
  hasToolUse: boolean
  /** 工具调用次数 */
  toolCount: number
  /** 是否包含图片 */
  hasImage: boolean
  /** 是否包含错误 */
  hasError: boolean
}

/** 轮次摘要 */
export interface TurnSummary {
  /** 用户消息摘要 */
  user?: string
  /** 助手消息摘要 */
  assistant?: string
}

/** 对话轮次 */
export interface ConversationTurn {
  /** 轮次序号（从 1 开始） */
  index: number
  /** 用户消息 */
  user?: ParsedMessage
  /** 助手回复 */
  assistant?: ParsedMessage
  /** 摘要文本（用于折叠预览） */
  summary: TurnSummary
  /** 统计信息 */
  stats: TurnStats
}

/** 分组后的对话结构 */
export interface GroupedConversation {
  /** 系统提示词 */
  system?: string
  /** 对话轮次列表 */
  turns: ConversationTurn[]
  /** 总轮次数 */
  totalTurns: number
  /** 是否为流式响应 */
  isStream: boolean
  /** 原始 API 格式 */
  apiFormat: ApiFormat
}
