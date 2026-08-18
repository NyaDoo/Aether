import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock } = vi.hoisted(() => ({ getMock: vi.fn(), postMock: vi.fn() }))

vi.mock('@/api/client', () => ({
  default: {
    get: getMock,
    post: postMock,
  },
}))

import { getProviderKeysPage, testAssetLibraryConnection } from '@/api/endpoints/keys'

describe('getProviderKeysPage', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
  })

  it('normalizes a legacy array response for the provider drawer', async () => {
    getMock.mockResolvedValue({
      data: [{ id: 'key-1' }, { id: 'key-2' }],
    })

    const result = await getProviderKeysPage('provider-demo', { page: 1, page_size: 1 })

    expect(result).toMatchObject({ total: 2, page: 1, page_size: 1 })
    expect(result.keys).toEqual([{ id: 'key-1' }])
  })

  it('normalizes a malformed object without exposing a non-array keys field', async () => {
    getMock.mockResolvedValue({
      data: { total: null, page: null, page_size: null, keys: {} },
    })

    const result = await getProviderKeysPage('provider-demo', { page: 2, page_size: 3 })

    expect(result).toEqual({ total: 0, page: 2, page_size: 3, keys: [] })
  })

  it('tests the exact saved key and asset-library endpoint without fallback', async () => {
    const payload = {
      success: true,
      action: 'ListAssetGroups',
      provider_id: 'provider-demo',
      endpoint_id: 'endpoint-asset',
      key_id: 'key/demo',
      status_code: 200,
      latency_ms: 32,
      request_id: 'req-1',
      total: 0,
    }
    postMock.mockResolvedValue({ data: payload })

    const controller = new AbortController()
    const result = await testAssetLibraryConnection('key/demo', 'endpoint-asset', {
      signal: controller.signal,
    })

    expect(postMock).toHaveBeenCalledWith(
      '/api/admin/endpoints/keys/key%2Fdemo/asset-library/test',
      { endpoint_id: 'endpoint-asset' },
      { timeout: 60_000, signal: controller.signal },
    )
    expect(result).toEqual(payload)
  })
})
