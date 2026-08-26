import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, cachedRequestMock, dedupedRequestMock, buildCacheKeyMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  cachedRequestMock: vi.fn(async (_key: string, fn: () => Promise<unknown>) => fn()),
  dedupedRequestMock: vi.fn(async (_key: string, fn: () => Promise<unknown>) => fn()),
  buildCacheKeyMock: vi.fn(() => 'cache-key'),
}))

vi.mock('@/api/client', () => ({
  default: {
    get: getMock,
  },
}))

vi.mock('@/utils/cache', () => ({
  cachedRequest: cachedRequestMock,
  dedupedRequest: dedupedRequestMock,
  buildCacheKey: buildCacheKeyMock,
}))

import { usageApi } from '@/api/usage'

describe('usageApi contract alignment', () => {
  beforeEach(() => {
    getMock.mockReset()
    cachedRequestMock.mockClear()
    dedupedRequestMock.mockClear()
    buildCacheKeyMock.mockClear()
  })

  it('loads current-user usage records from the Rust usage endpoint and normalizes pagination', async () => {
    getMock.mockResolvedValueOnce({
      data: {
        records: [{ id: 'record-1' }],
        pagination: {
          total: 42,
          limit: 10,
          offset: 10,
          has_more: true,
        },
      },
    })

    const result = await usageApi.getUsageRecords({
      page: 2,
      page_size: 10,
      start_date: '2026-05-01',
      end_date: '2026-05-16',
    })

    expect(getMock).toHaveBeenCalledWith('/api/users/me/usage', {
      params: {
        limit: 10,
        offset: 10,
        start_date: '2026-05-01',
        end_date: '2026-05-16',
      },
    })
    expect(result).toEqual({
      records: [{ id: 'record-1' }],
      total: 42,
      page: 2,
      page_size: 10,
    })
  })

  it('preserves minute-precision custom datetime bounds for current-user records', async () => {
    getMock.mockResolvedValueOnce({
      data: {
        records: [{ id: 'record-minute-boundary' }],
        pagination: {
          total: 1,
          limit: 20,
          offset: 0,
        },
      },
    })

    await usageApi.getUsageRecords({
      page: 1,
      page_size: 20,
      start_date: '2026-05-06T09:07',
      end_date: '2026-05-06T18:42',
      timezone: 'Asia/Shanghai',
      tz_offset_minutes: 480,
    })

    expect(getMock).toHaveBeenCalledWith('/api/users/me/usage', {
      params: {
        limit: 20,
        offset: 0,
        start_date: '2026-05-06T09:07',
        end_date: '2026-05-06T18:42',
        timezone: 'Asia/Shanghai',
        tz_offset_minutes: 480,
      },
    })
  })

  it('loads admin usage for a specific user from admin usage endpoints', async () => {
    getMock
      .mockResolvedValueOnce({
        data: {
          total_requests: 7,
          total_tokens: 99,
          total_cost: 12.34,
          avg_response_time: 456,
        },
      })
      .mockResolvedValueOnce({
        data: {
          records: [{ id: 'record-2' }],
          total: 1,
          limit: 25,
          offset: 0,
        },
      })

    const result = await usageApi.getUserUsage('user-123', {
      page: 1,
      page_size: 25,
      model: 'gpt-5.5',
    })

    expect(getMock).toHaveBeenNthCalledWith(1, '/api/admin/usage/stats', {
      params: {
        user_id: 'user-123',
        model: 'gpt-5.5',
      },
    })
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/admin/usage/records', {
      params: {
        user_id: 'user-123',
        limit: 25,
        offset: 0,
        model: 'gpt-5.5',
      },
    })
    expect(result).toEqual({
      records: [{ id: 'record-2' }],
      stats: {
        total_requests: 7,
        total_tokens: 99,
        total_cost: 12.34,
        avg_response_time: 456,
      },
    })
  })

  it('preserves minute-precision custom datetime bounds for admin records and stats', async () => {
    getMock
      .mockResolvedValueOnce({
        data: {
          total_requests: 1,
          total_tokens: 10,
          total_cost: 0.1,
          avg_response_time: 100,
        },
      })
      .mockResolvedValueOnce({
        data: {
          records: [{ id: 'admin-record-minute-boundary' }],
          total: 1,
          limit: 20,
          offset: 0,
        },
      })

    await usageApi.getUserUsage('user-123', {
      page: 1,
      page_size: 20,
      start_date: '2026-05-06T09:07',
      end_date: '2026-05-06T18:42',
      timezone: 'Asia/Shanghai',
      tz_offset_minutes: 480,
    })

    const expectedBounds = {
      start_date: '2026-05-06T09:07',
      end_date: '2026-05-06T18:42',
      timezone: 'Asia/Shanghai',
      tz_offset_minutes: 480,
    }
    expect(getMock).toHaveBeenNthCalledWith(1, '/api/admin/usage/stats', {
      params: {
        user_id: 'user-123',
        ...expectedBounds,
      },
    })
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/admin/usage/records', {
      params: {
        user_id: 'user-123',
        limit: 20,
        offset: 0,
        ...expectedBounds,
      },
    })
  })

  it('passes minute-precision bounds to the paged admin records and total queries', async () => {
    getMock
      .mockResolvedValueOnce({
        data: {
          records: [{ id: 'record-minute-page' }],
          total: 1,
          limit: 20,
          offset: 0,
        },
      })
      .mockResolvedValueOnce({ data: { total: 1 } })

    const bounds = {
      start_date: '2026-05-06T09:07',
      end_date: '2026-05-06T18:42',
      timezone: 'Asia/Shanghai',
      tz_offset_minutes: 480,
    }
    await usageApi.getAllUsageRecords({
      ...bounds,
      limit: 20,
      offset: 0,
      include_total: false,
    })
    await usageApi.getAllUsageRecordTotal(bounds)

    expect(getMock).toHaveBeenNthCalledWith(1, '/api/admin/usage/records', {
      params: {
        ...bounds,
        limit: 20,
        offset: 0,
        include_total: false,
      },
    })
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/admin/usage/records', {
      params: {
        ...bounds,
        include_total: true,
        total_only: true,
        limit: 1,
        offset: 0,
      },
    })
  })

  it('uses an extended timeout and cache bypass option for admin analytics', async () => {
    getMock
      .mockResolvedValueOnce({
        data: {
          total_requests: 7,
          total_tokens: 99,
          total_cost: 12.34,
          avg_response_time: 456,
        },
      })
      .mockResolvedValueOnce({
        data: [{ provider: 'OpenAI', request_count: 7 }],
      })

    await usageApi.getUsageStats({ preset: 'last30days' }, { skipCache: true })
    await usageApi.getUsageByProvider({ preset: 'last30days' }, { skipCache: true })

    expect(cachedRequestMock).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining(':fresh'),
      expect.any(Function),
      0
    )
    expect(cachedRequestMock).toHaveBeenNthCalledWith(
      2,
      expect.stringContaining(':fresh'),
      expect.any(Function),
      0
    )
    expect(getMock).toHaveBeenNthCalledWith(1, '/api/admin/usage/stats', {
      params: { preset: 'last30days' },
      timeout: 120000,
    })
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/admin/usage/aggregation/stats', {
      params: { group_by: 'provider', preset: 'last30days' },
      timeout: 120000,
    })
  })
})
