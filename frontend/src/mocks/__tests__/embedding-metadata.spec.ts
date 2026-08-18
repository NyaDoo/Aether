import { describe, expect, it } from 'vitest'

import { API_FORMATS } from '@/api/endpoints/types'
import { MOCK_API_FORMATS, MOCK_GLOBAL_MODELS } from '../data'

describe('embedding mock metadata', () => {
  it('keeps the mock API format catalog aligned with frontend format declarations', () => {
    expect(new Set(MOCK_API_FORMATS.formats.map(format => format.value))).toEqual(
      new Set(Object.values(API_FORMATS)),
    )
  })

  it('exposes embedding model metadata to frontend code without chat treatment', () => {
    const model = MOCK_GLOBAL_MODELS.find(item => item.name === 'text-embedding-3-small')

    expect(model).toMatchObject({
      supported_capabilities: ['embedding'],
      supports_embedding: true,
      config: {
        streaming: false,
        embedding: true,
        model_type: 'embedding',
        api_formats: ['openai:embedding'],
      },
    })
  })

  it('includes all embedding API formats as distinct catalog formats', () => {
    const embeddingFormats = MOCK_API_FORMATS.formats
      .filter(format => format.value.endsWith(':embedding') || format.value.endsWith('_embedding'))
      .map(format => [format.value, format.label])

    expect(embeddingFormats).toEqual([
      ['openai:embedding', 'OpenAI Embedding'],
      ['gemini:embedding', 'Gemini Embedding'],
      ['jina:embedding', 'Jina Embedding'],
      ['doubao:embedding', 'Doubao Embedding'],
      ['aliyun:multimodal_embedding', 'Aliyun Multimodal Embedding'],
    ])
  })

  it('includes rerank API formats as distinct catalog formats', () => {
    const rerankFormats = MOCK_API_FORMATS.formats
      .filter(format => format.value.endsWith(':rerank'))
      .map(format => [format.value, format.label])

    expect(rerankFormats).toEqual([
      ['openai:rerank', 'OpenAI Rerank'],
      ['jina:rerank', 'Jina Rerank'],
    ])
  })
})
