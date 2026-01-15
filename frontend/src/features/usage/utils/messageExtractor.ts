/**
 * 消息提取工具
 * 从不同 API 格式的请求体/响应体中提取人类可读的对话内容
 */

// ============================================================
// 类型定义
// ============================================================

export type ApiFormat = 'claude' | 'openai' | 'gemini' | 'unknown'
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool'
export type ContentType = 'text' | 'thinking' | 'tool_use' | 'tool_result' | 'image' | 'file'

export interface ExtractedMessage {
  role: MessageRole
  content: string
  type: ContentType
  metadata?: {
    toolName?: string
    toolId?: string
    fileName?: string
    mimeType?: string
  }
}

export interface ExtractedConversation {
  system?: string
  messages: ExtractedMessage[]
  isStream: boolean
  parseError?: string
}

// ============================================================
// API 格式检测
// ============================================================

export function detectApiFormat(
  requestBody: any,
  responseBody: any,
  apiFormatHint?: string
): ApiFormat {
  // 1. 优先使用后端提供的 api_format
  if (apiFormatHint) {
    const hint = apiFormatHint.toLowerCase()
    if (hint.includes('claude')) return 'claude'
    if (hint.includes('openai')) return 'openai'
    if (hint.includes('gemini')) return 'gemini'
  }

  // 2. 从请求体结构推断
  if (requestBody) {
    // Gemini: 使用 contents 而非 messages
    if (requestBody.contents && Array.isArray(requestBody.contents)) {
      return 'gemini'
    }
    // Claude vs OpenAI: 都有 messages，通过响应体区分
    if (requestBody.messages) {
      const respBody = isStreamResponse(responseBody)
        ? responseBody.chunks?.[0]
        : responseBody

      // Claude 响应特征: type="message" 或 content_block 事件
      if (
        respBody?.type === 'message' ||
        respBody?.type?.startsWith('content_block') ||
        respBody?.type?.startsWith('message_')
      ) {
        return 'claude'
      }
      // OpenAI 响应特征: choices 数组
      if (respBody?.choices || respBody?.object?.includes('chat.completion')) {
        return 'openai'
      }
      // 默认按 Claude 处理（Aether 主要用途）
      return 'claude'
    }
  }

  return 'unknown'
}

// ============================================================
// 流式响应检测
// ============================================================

export function isStreamResponse(body: any): boolean {
  return body?.metadata?.stream === true && Array.isArray(body?.chunks)
}

// ============================================================
// 请求体提取
// ============================================================

export function extractRequestMessages(
  requestBody: any,
  apiFormat: ApiFormat
): ExtractedConversation {
  if (!requestBody) {
    return { messages: [], isStream: false, parseError: '无请求体' }
  }

  try {
    switch (apiFormat) {
      case 'claude':
        return extractClaudeRequest(requestBody)
      case 'openai':
        return extractOpenAIRequest(requestBody)
      case 'gemini':
        return extractGeminiRequest(requestBody)
      default:
        return { messages: [], isStream: false, parseError: '无法识别的 API 格式' }
    }
  } catch (e) {
    return { messages: [], isStream: false, parseError: `解析失败: ${e}` }
  }
}

// ============================================================
// 响应体提取
// ============================================================

export function extractResponseMessages(
  responseBody: any,
  apiFormat: ApiFormat
): ExtractedConversation {
  if (!responseBody) {
    return { messages: [], isStream: false, parseError: '无响应体' }
  }

  const isStream = isStreamResponse(responseBody)

  try {
    switch (apiFormat) {
      case 'claude':
        return isStream
          ? extractClaudeStreamResponse(responseBody.chunks)
          : extractClaudeResponse(responseBody)
      case 'openai':
        return isStream
          ? extractOpenAIStreamResponse(responseBody.chunks)
          : extractOpenAIResponse(responseBody)
      case 'gemini':
        return isStream
          ? extractGeminiStreamResponse(responseBody.chunks)
          : extractGeminiResponse(responseBody)
      default:
        return { messages: [], isStream, parseError: '无法识别的 API 格式' }
    }
  } catch (e) {
    return { messages: [], isStream, parseError: `解析失败: ${e}` }
  }
}

// ============================================================
// Claude 格式提取
// ============================================================

function extractClaudeRequest(body: any): ExtractedConversation {
  const result: ExtractedConversation = { messages: [], isStream: false }

  // 提取 system prompt
  if (body.system) {
    if (typeof body.system === 'string') {
      result.system = body.system
    } else if (Array.isArray(body.system)) {
      result.system = body.system
        .filter((b: any) => b.type === 'text')
        .map((b: any) => b.text)
        .join('\n')
    }
  }

  // 提取 messages
  if (Array.isArray(body.messages)) {
    for (const msg of body.messages) {
      const role = msg.role as MessageRole

      if (typeof msg.content === 'string') {
        result.messages.push({ role, content: msg.content, type: 'text' })
      } else if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block.type === 'text') {
            result.messages.push({ role, content: block.text, type: 'text' })
          } else if (block.type === 'image') {
            result.messages.push({
              role,
              content: '[图片]',
              type: 'image',
              metadata: { mimeType: block.source?.media_type },
            })
          } else if (block.type === 'tool_use') {
            result.messages.push({
              role,
              content: JSON.stringify(block.input, null, 2),
              type: 'tool_use',
              metadata: { toolName: block.name, toolId: block.id },
            })
          } else if (block.type === 'tool_result') {
            const content =
              typeof block.content === 'string'
                ? block.content
                : JSON.stringify(block.content, null, 2)
            result.messages.push({
              role,
              content,
              type: 'tool_result',
              metadata: { toolId: block.tool_use_id },
            })
          } else if (block.type === 'thinking') {
            result.messages.push({ role, content: block.thinking, type: 'thinking' })
          }
        }
      }
    }
  }

  return result
}

function extractClaudeResponse(body: any): ExtractedConversation {
  const result: ExtractedConversation = { messages: [], isStream: false }

  if (Array.isArray(body.content)) {
    for (const block of body.content) {
      if (block.type === 'text') {
        result.messages.push({ role: 'assistant', content: block.text, type: 'text' })
      } else if (block.type === 'thinking') {
        result.messages.push({ role: 'assistant', content: block.thinking, type: 'thinking' })
      } else if (block.type === 'tool_use') {
        result.messages.push({
          role: 'assistant',
          content: JSON.stringify(block.input, null, 2),
          type: 'tool_use',
          metadata: { toolName: block.name, toolId: block.id },
        })
      }
    }
  }

  return result
}

function extractClaudeStreamResponse(chunks: any[]): ExtractedConversation {
  const result: ExtractedConversation = { messages: [], isStream: true }

  // 按 content block index 分组累积
  const blocks: Map<number, { type: ContentType; parts: string[]; metadata?: any }> = new Map()

  for (const chunk of chunks) {
    if (chunk.type === 'content_block_start') {
      const index = chunk.index ?? 0
      const block = chunk.content_block
      if (block?.type === 'text') {
        blocks.set(index, { type: 'text', parts: [block.text || ''] })
      } else if (block?.type === 'thinking') {
        blocks.set(index, { type: 'thinking', parts: [block.thinking || ''] })
      } else if (block?.type === 'tool_use') {
        blocks.set(index, {
          type: 'tool_use',
          parts: [],
          metadata: { toolName: block.name, toolId: block.id },
        })
      }
    } else if (chunk.type === 'content_block_delta') {
      const index = chunk.index ?? 0
      const delta = chunk.delta
      const block = blocks.get(index)
      if (block) {
        if (delta?.type === 'text_delta') {
          block.parts.push(delta.text || '')
        } else if (delta?.type === 'thinking_delta') {
          block.parts.push(delta.thinking || '')
        } else if (delta?.type === 'input_json_delta') {
          block.parts.push(delta.partial_json || '')
        }
      }
    }
  }

  // 转换为消息
  for (const [index, block] of Array.from(blocks.entries()).sort((a, b) => a[0] - b[0])) {
    result.messages.push({
      role: 'assistant',
      content: block.parts.join(''),
      type: block.type,
      metadata: block.metadata,
    })
  }

  return result
}

// ============================================================
// OpenAI 格式提取
// ============================================================

function extractOpenAIRequest(body: any): ExtractedConversation {
  const result: ExtractedConversation = { messages: [], isStream: false }

  if (Array.isArray(body.messages)) {
    for (const msg of body.messages) {
      const role = msg.role as MessageRole

      if (role === 'system') {
        result.system = (result.system || '') + (typeof msg.content === 'string' ? msg.content : '')
        continue
      }

      if (typeof msg.content === 'string') {
        result.messages.push({ role, content: msg.content, type: 'text' })
      } else if (Array.isArray(msg.content)) {
        // Vision API 格式
        for (const part of msg.content) {
          if (part.type === 'text') {
            result.messages.push({ role, content: part.text, type: 'text' })
          } else if (part.type === 'image_url') {
            result.messages.push({ role, content: '[图片]', type: 'image' })
          }
        }
      }

      // 工具调用
      if (msg.tool_calls) {
        for (const call of msg.tool_calls) {
          result.messages.push({
            role,
            content: call.function?.arguments || '{}',
            type: 'tool_use',
            metadata: { toolName: call.function?.name, toolId: call.id },
          })
        }
      }

      // 工具结果
      if (msg.tool_call_id) {
        result.messages.push({
          role: 'tool',
          content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content),
          type: 'tool_result',
          metadata: { toolId: msg.tool_call_id },
        })
      }
    }
  }

  return result
}

function extractOpenAIResponse(body: any): ExtractedConversation {
  const result: ExtractedConversation = { messages: [], isStream: false }

  const message = body.choices?.[0]?.message
  if (message) {
    if (message.content) {
      result.messages.push({ role: 'assistant', content: message.content, type: 'text' })
    }
    if (message.tool_calls) {
      for (const call of message.tool_calls) {
        result.messages.push({
          role: 'assistant',
          content: call.function?.arguments || '{}',
          type: 'tool_use',
          metadata: { toolName: call.function?.name, toolId: call.id },
        })
      }
    }
  }

  return result
}

function extractOpenAIStreamResponse(chunks: any[]): ExtractedConversation {
  const result: ExtractedConversation = { messages: [], isStream: true }
  const textParts: string[] = []
  const toolCalls: Map<number, { name: string; id: string; args: string[] }> = new Map()

  for (const chunk of chunks) {
    const delta = chunk.choices?.[0]?.delta
    if (delta?.content) {
      textParts.push(delta.content)
    }
    if (delta?.tool_calls) {
      for (const call of delta.tool_calls) {
        const index = call.index ?? 0
        if (!toolCalls.has(index)) {
          toolCalls.set(index, { name: call.function?.name || '', id: call.id || '', args: [] })
        }
        if (call.function?.arguments) {
          toolCalls.get(index)!.args.push(call.function.arguments)
        }
      }
    }
  }

  if (textParts.length) {
    result.messages.push({ role: 'assistant', content: textParts.join(''), type: 'text' })
  }

  for (const [, call] of toolCalls) {
    result.messages.push({
      role: 'assistant',
      content: call.args.join(''),
      type: 'tool_use',
      metadata: { toolName: call.name, toolId: call.id },
    })
  }

  return result
}

// ============================================================
// Gemini 格式提取
// ============================================================

function extractGeminiRequest(body: any): ExtractedConversation {
  const result: ExtractedConversation = { messages: [], isStream: false }

  // System instruction
  if (body.system_instruction || body.systemInstruction) {
    const sysInst = body.system_instruction || body.systemInstruction
    if (sysInst.parts) {
      result.system = sysInst.parts
        .filter((p: any) => p.text)
        .map((p: any) => p.text)
        .join('\n')
    }
  }

  // Contents
  if (Array.isArray(body.contents)) {
    for (const content of body.contents) {
      const role = content.role === 'model' ? 'assistant' : ((content.role || 'user') as MessageRole)

      if (Array.isArray(content.parts)) {
        for (const part of content.parts) {
          if (part.text) {
            result.messages.push({ role, content: part.text, type: 'text' })
          } else if (part.inlineData) {
            result.messages.push({
              role,
              content: '[图片]',
              type: 'image',
              metadata: { mimeType: part.inlineData.mimeType },
            })
          } else if (part.functionCall) {
            result.messages.push({
              role,
              content: JSON.stringify(part.functionCall.args, null, 2),
              type: 'tool_use',
              metadata: { toolName: part.functionCall.name },
            })
          } else if (part.functionResponse) {
            result.messages.push({
              role,
              content: JSON.stringify(part.functionResponse.response, null, 2),
              type: 'tool_result',
              metadata: { toolName: part.functionResponse.name },
            })
          }
        }
      }
    }
  }

  return result
}

function extractGeminiResponse(body: any): ExtractedConversation {
  const result: ExtractedConversation = { messages: [], isStream: false }

  const candidate = body.candidates?.[0]
  if (candidate?.content?.parts) {
    for (const part of candidate.content.parts) {
      if (part.text) {
        result.messages.push({ role: 'assistant', content: part.text, type: 'text' })
      } else if (part.functionCall) {
        result.messages.push({
          role: 'assistant',
          content: JSON.stringify(part.functionCall.args, null, 2),
          type: 'tool_use',
          metadata: { toolName: part.functionCall.name },
        })
      }
    }
  }

  return result
}

function extractGeminiStreamResponse(chunks: any[]): ExtractedConversation {
  const result: ExtractedConversation = { messages: [], isStream: true }
  const textParts: string[] = []
  const toolCalls: { name: string; args: any }[] = []

  for (const chunk of chunks) {
    const parts = chunk.candidates?.[0]?.content?.parts
    if (parts) {
      for (const part of parts) {
        if (part.text) {
          textParts.push(part.text)
        } else if (part.functionCall) {
          toolCalls.push(part.functionCall)
        }
      }
    }
  }

  if (textParts.length) {
    result.messages.push({ role: 'assistant', content: textParts.join(''), type: 'text' })
  }

  for (const call of toolCalls) {
    result.messages.push({
      role: 'assistant',
      content: JSON.stringify(call.args, null, 2),
      type: 'tool_use',
      metadata: { toolName: call.name },
    })
  }

  return result
}

// ============================================================
// 格式化输出
// ============================================================

export function formatConversationAsText(conversation: ExtractedConversation): string {
  const lines: string[] = []

  if (conversation.system) {
    lines.push('=== System ===')
    lines.push(conversation.system)
    lines.push('')
  }

  for (const msg of conversation.messages) {
    const roleLabel = msg.role.charAt(0).toUpperCase() + msg.role.slice(1)
    let header = `=== ${roleLabel} ===`

    if (msg.type === 'thinking') {
      header = `=== ${roleLabel} (Thinking) ===`
    } else if (msg.type === 'tool_use') {
      header = `=== ${roleLabel} (Tool: ${msg.metadata?.toolName || 'unknown'}) ===`
    } else if (msg.type === 'tool_result') {
      header = `=== Tool Result ===`
    }

    lines.push(header)
    lines.push(msg.content)
    lines.push('')
  }

  return lines.join('\n').trim()
}
