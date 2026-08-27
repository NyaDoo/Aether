import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, cachedRequestMock, buildCacheKeyMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  cachedRequestMock: vi.fn(),
  buildCacheKeyMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  default: {
    get: getMock,
  },
}))

// The realtime endpoint must not use the dashboard's historical response
// cache. Keep the cache helpers mocked so this contract test fails if the
// implementation accidentally routes realtime reads through cachedRequest.
vi.mock('@/utils/cache', () => ({
  cachedRequest: cachedRequestMock,
  buildCacheKey: buildCacheKeyMock,
}))

import { dashboardApi } from '@/api/dashboard'

describe('dashboard realtime metrics API contract', () => {
  beforeEach(() => {
    getMock.mockReset()
    cachedRequestMock.mockReset()
    buildCacheKeyMock.mockReset()
  })

  it('reads the shared realtime endpoint without the historical cache', async () => {
    const payload = {
      rpm: 12,
      tpm: 4_096,
      window_seconds: 60,
      as_of: '2026-08-27T12:00:00Z',
      semantics: {
        rpm: 'accepted_non_failed_requests',
        tpm: 'observed_token_deltas_including_failed',
        window: 'trailing_60_seconds',
        failed_requests: 'excluded_from_rpm_only',
      },
      storage_scope: 'shared',
    }
    getMock
      .mockResolvedValueOnce({ data: payload })
      .mockResolvedValueOnce({ data: { ...payload, rpm: 13, tpm: 5_120 } })

    await expect(dashboardApi.getRealtimeMetrics()).resolves.toEqual(payload)
    await expect(dashboardApi.getRealtimeMetrics()).resolves.toMatchObject({
      rpm: 13,
      tpm: 5_120,
    })

    expect(getMock).toHaveBeenCalledTimes(2)
    expect(getMock).toHaveBeenNthCalledWith(1, '/api/dashboard/realtime')
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/dashboard/realtime')
    expect(cachedRequestMock).not.toHaveBeenCalled()
    expect(buildCacheKeyMock).not.toHaveBeenCalled()
  })

  it('preserves the server-declared semantics and storage scope', async () => {
    const payload = {
      rpm: 0,
      tpm: 0,
      window_seconds: 60,
      as_of: '2026-08-27T12:00:01.250Z',
      semantics: {
        rpm: 'accepted_non_failed_requests',
        tpm: 'observed_token_deltas_including_failed',
        window: 'trailing_60_seconds',
        failed_requests: 'excluded_from_rpm_only',
      },
      storage_scope: 'process',
    }
    getMock.mockResolvedValueOnce({ data: payload })

    const result = await dashboardApi.getRealtimeMetrics()

    expect(result.window_seconds).toBe(60)
    expect(result.semantics).toEqual({
      rpm: 'accepted_non_failed_requests',
      tpm: 'observed_token_deltas_including_failed',
      window: 'trailing_60_seconds',
      failed_requests: 'excluded_from_rpm_only',
    })
    expect(result.storage_scope).toBe('process')
    expect(result.as_of).toBe(payload.as_of)
  })
})
