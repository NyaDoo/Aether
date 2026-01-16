/**
 * 对话轮次分组器
 * 将消息列表按轮次分组，生成摘要和统计信息
 */

import type {
  ParsedMessage,
  ParsedConversation,
  ContentBlock,
  ConversationTurn,
  GroupedConversation,
  TurnStats,
  TurnSummary,
} from './types'

// ============================================================
// 常量配置
// ============================================================

/** 摘要最大长度 */
const SUMMARY_MAX_LENGTH = 50

/** 摘要截断时的最小保留比例 */
const SUMMARY_MIN_RATIO = 0.6

// ============================================================
// 摘要生成
// ============================================================

/**
 * 从内容块中提取纯文本
 */
function extractTextFromBlocks(blocks: ContentBlock[]): string {
  return blocks
    .filter((b): b is ContentBlock & { type: 'text' } => b.type === 'text')
    .map(b => (b as any).text || '')
    .join(' ')
    .trim()
}

/**
 * 生成摘要文本
 * 智能截断，在词边界或标点处断开
 */
function generateSummary(text: string, maxLength: number = SUMMARY_MAX_LENGTH): string {
  if (!text) return ''

  // 移除多余空白
  const normalized = text.replace(/\s+/g, ' ').trim()

  if (normalized.length <= maxLength) {
    return normalized
  }

  // 截取前 maxLength 个字符
  let truncated = normalized.slice(0, maxLength)

  // 尝试在词边界或标点处断开
  const lastSpace = truncated.lastIndexOf(' ')
  const lastPunctCN = Math.max(
    truncated.lastIndexOf('，'),
    truncated.lastIndexOf('。'),
    truncated.lastIndexOf('、'),
    truncated.lastIndexOf('；')
  )
  const lastPunctEN = Math.max(
    truncated.lastIndexOf(','),
    truncated.lastIndexOf('.'),
    truncated.lastIndexOf(';')
  )

  const cutPoint = Math.max(lastSpace, lastPunctCN, lastPunctEN)
  const minCutPoint = maxLength * SUMMARY_MIN_RATIO

  if (cutPoint > minCutPoint) {
    truncated = truncated.slice(0, cutPoint)
  }

  return `${truncated}...`
}

/**
 * 生成消息摘要
 * 优先提取文本，如果没有文本则根据内容类型生成描述
 */
function generateMessageSummary(message: ParsedMessage | undefined): string | undefined {
  if (!message || message.content.length === 0) {
    return undefined
  }

  // 优先提取文本内容
  const text = extractTextFromBlocks(message.content)
  if (text) {
    return generateSummary(text)
  }

  // 没有文本时，根据内容类型生成描述
  const hasThinking = message.content.some(b => b.type === 'thinking')
  const hasToolUse = message.content.some(b => b.type === 'tool_use')
  const hasToolResult = message.content.some(b => b.type === 'tool_result')
  const hasImage = message.content.some(b => b.type === 'image')
  const hasError = message.content.some(b => b.type === 'error')

  const parts: string[] = []
  if (hasThinking) parts.push('思考过程')
  if (hasToolUse) parts.push('工具调用')
  if (hasToolResult) parts.push('工具结果')
  if (hasImage) parts.push('图片')
  if (hasError) parts.push('错误')

  return parts.length > 0 ? `[${parts.join(', ')}]` : undefined
}

// ============================================================
// 统计信息计算
// ============================================================

/**
 * 计算消息的字符数
 */
function countMessageChars(message: ParsedMessage | undefined): number {
  if (!message) return 0

  return message.content.reduce((total, block) => {
    switch (block.type) {
      case 'text':
        return total + (block.text?.length || 0)
      case 'thinking':
        return total + (block.thinking?.length || 0)
      case 'code':
        return total + (block.code?.length || 0)
      default:
        return total
    }
  }, 0)
}

/**
 * 计算轮次统计信息
 */
function calculateTurnStats(
  userMessage: ParsedMessage | undefined,
  assistantMessage: ParsedMessage | undefined
): TurnStats {
  const allBlocks = [
    ...(userMessage?.content || []),
    ...(assistantMessage?.content || []),
  ]

  const toolUseBlocks = allBlocks.filter(b => b.type === 'tool_use')

  return {
    userChars: countMessageChars(userMessage),
    assistantChars: countMessageChars(assistantMessage),
    hasThinking: allBlocks.some(b => b.type === 'thinking'),
    hasToolUse: toolUseBlocks.length > 0,
    toolCount: toolUseBlocks.length,
    hasImage: allBlocks.some(b => b.type === 'image'),
    hasError: allBlocks.some(b => b.type === 'error'),
  }
}

// ============================================================
// 轮次分组
// ============================================================

/**
 * 将消息列表按轮次分组
 *
 * 分组规则：
 * 1. 每个 user 消息开始一个新轮次
 * 2. 紧随其后的 assistant 消息属于同一轮次
 * 3. 连续的 assistant 消息合并到同一轮次
 * 4. tool 消息归属于前一个 assistant 的轮次
 */
export function groupMessagesIntoTurns(messages: ParsedMessage[]): ConversationTurn[] {
  const turns: ConversationTurn[] = []
  let currentTurn: Partial<ConversationTurn> | null = null
  let turnIndex = 0

  for (const message of messages) {
    if (message.role === 'user') {
      // 保存之前的轮次
      if (currentTurn) {
        turns.push(finalizeTurn(currentTurn, turnIndex))
        turnIndex++
      }

      // 开始新轮次
      currentTurn = {
        user: message,
        assistant: undefined,
      }
    } else if (message.role === 'assistant') {
      if (!currentTurn) {
        // 没有 user 消息的 assistant（可能是响应体）
        currentTurn = {
          user: undefined,
          assistant: message,
        }
      } else if (currentTurn.assistant) {
        // 已有 assistant，合并内容
        currentTurn.assistant = {
          role: 'assistant',
          content: [...currentTurn.assistant.content, ...message.content],
        }
      } else {
        currentTurn.assistant = message
      }
    } else if (message.role === 'tool') {
      // tool 消息归属于当前轮次的 assistant
      if (currentTurn?.assistant) {
        currentTurn.assistant = {
          role: 'assistant',
          content: [...currentTurn.assistant.content, ...message.content],
        }
      }
    }
    // system 消息在分组前已被提取，这里忽略
  }

  // 保存最后一个轮次
  if (currentTurn) {
    turns.push(finalizeTurn(currentTurn, turnIndex))
  }

  return turns
}

/**
 * 完成轮次构建，添加摘要和统计信息
 */
function finalizeTurn(
  partial: Partial<ConversationTurn>,
  index: number
): ConversationTurn {
  const summary: TurnSummary = {
    user: generateMessageSummary(partial.user),
    assistant: generateMessageSummary(partial.assistant),
  }

  const stats = calculateTurnStats(partial.user, partial.assistant)

  return {
    index: index + 1, // 从 1 开始
    user: partial.user,
    assistant: partial.assistant,
    summary,
    stats,
  }
}

// ============================================================
// 主入口
// ============================================================

/**
 * 将解析后的对话转换为分组对话
 */
export function groupConversation(conversation: ParsedConversation): GroupedConversation {
  // 过滤掉 system 消息（system 已在 conversation.system 中）
  const nonSystemMessages = conversation.messages.filter(m => m.role !== 'system')

  const turns = groupMessagesIntoTurns(nonSystemMessages)

  return {
    system: conversation.system,
    turns,
    totalTurns: turns.length,
    isStream: conversation.isStream,
    apiFormat: conversation.apiFormat,
  }
}

/**
 * 从 RenderResult 的 blocks 中提取轮次
 * 用于已渲染的结果进行轮次分组
 */
export function groupRenderBlocksIntoTurns(
  blocks: import('./render').RenderBlock[],
  isStream: boolean = false
): GroupedConversation {
  const turns: ConversationTurn[] = []
  let currentTurn: Partial<ConversationTurn> | null = null
  let turnIndex = 0
  let systemPrompt: string | undefined

  for (const block of blocks) {
    if (block.type !== 'message') continue

    const messageBlock = block as import('./render').MessageRenderBlock

    if (messageBlock.role === 'system') {
      // 提取 system prompt
      systemPrompt = extractTextFromRenderBlocks(messageBlock.content)
      continue
    }

    if (messageBlock.role === 'user') {
      // 保存之前的轮次
      if (currentTurn) {
        turns.push(finalizeTurnFromRenderBlocks(currentTurn, turnIndex))
        turnIndex++
      }

      // 开始新轮次
      currentTurn = {
        user: renderBlockToMessage(messageBlock),
        assistant: undefined,
      }
    } else if (messageBlock.role === 'assistant') {
      if (!currentTurn) {
        currentTurn = {
          user: undefined,
          assistant: renderBlockToMessage(messageBlock),
        }
      } else if (currentTurn.assistant) {
        // 合并 assistant 内容
        const existingContent = currentTurn.assistant.content
        const newContent = renderBlockToMessage(messageBlock).content
        currentTurn.assistant = {
          role: 'assistant',
          content: [...existingContent, ...newContent],
        }
      } else {
        currentTurn.assistant = renderBlockToMessage(messageBlock)
      }
    }
  }

  // 保存最后一个轮次
  if (currentTurn) {
    turns.push(finalizeTurnFromRenderBlocks(currentTurn, turnIndex))
  }

  return {
    system: systemPrompt,
    turns,
    totalTurns: turns.length,
    isStream,
    apiFormat: 'unknown',
  }
}

/**
 * 从渲染块中提取文本
 */
function extractTextFromRenderBlocks(blocks: import('./render').RenderBlock[]): string {
  return blocks
    .filter(b => b.type === 'text')
    .map(b => (b as any).content || '')
    .join(' ')
    .trim()
}

/**
 * 将 MessageRenderBlock 转换为 ParsedMessage
 */
function renderBlockToMessage(block: import('./render').MessageRenderBlock): ParsedMessage {
  const content: ContentBlock[] = []

  for (const child of block.content) {
    switch (child.type) {
      case 'text':
        content.push({ type: 'text', text: (child as any).content || '' })
        break
      case 'collapsible':
        // 可能是 thinking，从 code 块中提取内容
        if ((child as any).title?.includes('思考')) {
          const innerBlocks = (child as any).content || []
          const codeBlock = innerBlocks.find((b: any) => b.type === 'code')
          const thinkingText = codeBlock?.code || ''
          content.push({ type: 'thinking', thinking: thinkingText })
        }
        break
      case 'tool_use':
        content.push({
          type: 'tool_use',
          toolId: (child as any).toolId || '',
          toolName: (child as any).toolName || '',
          input: (child as any).input || '',
        })
        break
      case 'tool_result':
        content.push({
          type: 'tool_result',
          toolUseId: '',
          content: (child as any).content || '',
          isError: (child as any).isError,
        })
        break
      case 'image':
        content.push({
          type: 'image',
          sourceType: 'url',
          url: (child as any).src,
          mimeType: (child as any).mimeType,
          alt: (child as any).alt,
        })
        break
      case 'error':
        content.push({
          type: 'error',
          message: (child as any).message || '',
          code: (child as any).code,
        })
        break
    }
  }

  return {
    role: block.role as any,
    content,
  }
}

/**
 * 从渲染块构建的轮次完成
 */
function finalizeTurnFromRenderBlocks(
  partial: Partial<ConversationTurn>,
  index: number
): ConversationTurn {
  return finalizeTurn(partial, index)
}
