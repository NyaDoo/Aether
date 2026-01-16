/**
 * 对话解析器
 *
 * 统一解析 Claude/OpenAI/Gemini 等 API 格式的请求和响应
 *
 * @example
 * ```ts
 * import { renderRequest, renderResponse, detectApiFormat } from '@/features/usage/conversation'
 *
 * // 检测 API 格式
 * const format = detectApiFormat(requestBody, responseBody, apiFormatHint)
 *
 * // 渲染请求体为渲染块
 * const requestResult = renderRequest(requestBody, responseBody, apiFormatHint)
 *
 * // 渲染响应体为渲染块
 * const responseResult = renderResponse(responseBody, requestBody, apiFormatHint)
 * ```
 */

// 导出解析类型
export type {
  ApiFormat,
  MessageRole,
  ContentBlockType,
  ContentBlock,
  TextContentBlock,
  ThinkingContentBlock,
  ToolUseContentBlock,
  ToolResultContentBlock,
  ImageContentBlock,
  FileContentBlock,
  CodeContentBlock,
  ErrorContentBlock,
  ParsedMessage,
  ParsedConversation,
  ApiFormatParser,
  FormatDetector,
  RequestParser,
  ResponseParser,
  // 轮次相关类型
  TurnStats,
  TurnSummary,
  ConversationTurn,
  GroupedConversation,
} from './types'

// 导出渲染类型
export type {
  RenderBlock,
  RenderResult,
  TextRenderBlock,
  CollapsibleRenderBlock,
  CodeRenderBlock,
  BadgeRenderBlock,
  ImageRenderBlock,
  ErrorRenderBlock,
  ContainerRenderBlock,
  MessageRenderBlock,
  ToolUseRenderBlock,
  ToolResultRenderBlock,
  DividerRenderBlock,
  LabelRenderBlock,
} from './render'

// 导出解析工具函数
export {
  isStreamResponse,
  createTextBlock,
  createThinkingBlock,
  createToolUseBlock,
  createToolResultBlock,
  createImageBlock,
  createErrorBlock,
  createEmptyConversation,
  createMessage,
} from './types'

// 导出渲染工具函数
export {
  createTextBlock as createTextRenderBlock,
  createCollapsibleBlock,
  createCodeBlock,
  createBadgeBlock,
  createImageBlock as createImageRenderBlock,
  createErrorBlock as createErrorRenderBlock,
  createContainerBlock,
  createMessageBlock,
  createToolUseBlock as createToolUseRenderBlock,
  createToolResultBlock as createToolResultRenderBlock,
  createDividerBlock,
  createLabelBlock,
  createEmptyRenderResult,
} from './render'

// 导出解析器
export { claudeParser, ClaudeParser } from './claude'
export { openaiParser, OpenAIParser } from './openai'
export { geminiParser, GeminiParser } from './gemini'

// 导出注册表和统一入口
export {
  parserRegistry,
  parseRequest,
  parseResponse,
  detectApiFormat,
  renderRequest,
  renderResponse,
} from './registry'

// 导出轮次分组
export {
  groupMessagesIntoTurns,
  groupConversation,
  groupRenderBlocksIntoTurns,
} from './grouper'

// 导出转换器
export {
  contentBlockToRenderBlock,
  contentBlocksToRenderBlocks,
} from './converter'
