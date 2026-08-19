import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  default: {
    get: getMock,
    post: postMock,
  },
}))

import { createVideoTasksApi, videoTasksApi } from '@/api/video-tasks'

describe('video tasks API scopes', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
  })

  it('uses the self-service surface and drops user impersonation filters', async () => {
    getMock.mockResolvedValue({
      data: { items: [], total: 0, page: 2, page_size: 10, pages: 0 },
    })

    await createVideoTasksApi('user').list({
      status: 'processing',
      user_id: 'other-user',
      model: 'seedance pro',
      page: 2,
      page_size: 10,
    })

    expect(getMock).toHaveBeenCalledWith(
      '/api/users/me/video-tasks?status=processing&model=seedance+pro&page=2&page_size=10',
    )
  })

  it('keeps the existing admin client and explicit owner filtering', async () => {
    getMock.mockResolvedValue({
      data: { items: [], total: 0, page: 1, page_size: 20, pages: 0 },
    })

    await videoTasksApi.list({ user_id: 'user-42' })

    expect(getMock).toHaveBeenCalledWith('/api/admin/video-tasks?user_id=user-42')
  })

  it('encodes task ids and loads user media as an authenticated blob request', async () => {
    const video = new Blob(['video'], { type: 'video/mp4' })
    const controller = new AbortController()
    getMock.mockResolvedValue({ data: video })

    await expect(
      createVideoTasksApi('user').getVideoBlob('task/id', controller.signal),
    ).resolves.toBe(video)

    expect(getMock).toHaveBeenCalledWith(
      '/api/users/me/video-tasks/task%2Fid/video',
      { responseType: 'blob', signal: controller.signal },
    )
  })

  it('keeps query-token media URLs confined to the admin scope', () => {
    expect(createVideoTasksApi('admin').videoUrl('task/id', 'admin token')).toBe(
      '/api/admin/video-tasks/task%2Fid/video?token=admin%20token',
    )
    expect(createVideoTasksApi('user').videoUrl('task/id', 'user token')).toBe(
      '/api/users/me/video-tasks/task%2Fid/video',
    )
  })

  it('uses the scoped cancel endpoint', async () => {
    postMock.mockResolvedValue({ data: { id: 'task/id', status: 'cancelled', message: 'ok' } })

    await createVideoTasksApi('user').cancel('task/id')

    expect(postMock).toHaveBeenCalledWith('/api/users/me/video-tasks/task%2Fid/cancel')
  })
})
