/**
 * ContentBlock 到 RenderBlock 的转换器
 */

import type { ContentBlock } from './types'
import type { RenderBlock } from './render'
import {
  createTextBlock,
  createCollapsibleBlock,
  createCodeBlock,
  createToolUseBlock,
  createToolResultBlock,
  createImageBlock,
  createErrorBlock,
} from './render'

/**
 * 将单个 ContentBlock 转换为 RenderBlock
 */
export function contentBlockToRenderBlock(block: ContentBlock): RenderBlock | null {
  switch (block.type) {
    case 'text':
      return createTextBlock(block.text)

    case 'thinking':
      return createCollapsibleBlock(
        `思考过程 (${block.thinking.length} 字符)`,
        [createCodeBlock(block.thinking)],
        { defaultOpen: false }
      )

    case 'tool_use': {
      const input = typeof block.input === 'string'
        ? block.input
        : JSON.stringify(block.input, null, 2)
      return createToolUseBlock(block.toolName, input, block.toolId)
    }

    case 'tool_result': {
      const content = typeof block.content === 'string'
        ? block.content
        : JSON.stringify(block.content, null, 2)
      return createToolResultBlock(content, block.isError)
    }

    case 'image':
      return createImageBlock({
        src: block.sourceType === 'base64'
          ? `data:${block.mimeType || 'image/png'};base64,${block.data}`
          : block.url,
        mimeType: block.mimeType,
        alt: block.alt,
      })

    case 'error':
      return createErrorBlock(block.message, block.code)

    case 'code':
      return createCodeBlock(block.code, block.language)

    default:
      return null
  }
}

/**
 * 将 ContentBlock 数组转换为 RenderBlock 数组
 */
export function contentBlocksToRenderBlocks(blocks: ContentBlock[]): RenderBlock[] {
  return blocks
    .map(contentBlockToRenderBlock)
    .filter((b): b is RenderBlock => b !== null)
}
