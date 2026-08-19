import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createApp, h, nextTick, type App } from 'vue'

const { createVideoTasksApiMock, taskApiMocks, toastMock } = vi.hoisted(() => ({
  createVideoTasksApiMock: vi.fn(),
  taskApiMocks: {
    list: vi.fn(),
    getStats: vi.fn(),
    getDetail: vi.fn(),
    cancel: vi.fn(),
    getVideoBlob: vi.fn(),
    videoUrl: vi.fn(),
  },
  toastMock: vi.fn(),
}))

vi.mock('@/api/video-tasks', () => ({
  createVideoTasksApi: createVideoTasksApiMock,
}))

vi.mock('@/composables/useToast', () => ({
  useToast: () => ({ toast: toastMock }),
}))

vi.mock('@/features/usage/components', async () => {
  const { defineComponent, h } = await import('vue')
  return {
    RequestDetailDrawer: defineComponent({
      name: 'RequestDetailDrawerStub',
      setup: () => () => h('div'),
    }),
  }
})

import VideoTasksPanel from '@/features/async-tasks/components/VideoTasksPanel.vue'

const mountedApps: Array<{ app: App, root: HTMLElement }> = []
const createObjectUrlMock = vi.fn(() => 'blob:aether-video')
const revokeObjectUrlMock = vi.fn()

async function settle() {
  for (let index = 0; index < 8; index += 1) {
    await Promise.resolve()
    await nextTick()
  }
}

beforeEach(() => {
  for (const mock of Object.values(taskApiMocks)) mock.mockReset()
  createVideoTasksApiMock.mockReset()
  toastMock.mockReset()
  createObjectUrlMock.mockClear()
  revokeObjectUrlMock.mockClear()

  Object.defineProperty(URL, 'createObjectURL', {
    value: createObjectUrlMock,
    configurable: true,
  })
  Object.defineProperty(URL, 'revokeObjectURL', {
    value: revokeObjectUrlMock,
    configurable: true,
  })

  createVideoTasksApiMock.mockReturnValue(taskApiMocks)
  taskApiMocks.list.mockResolvedValue({
    items: [{
      id: 'task-1',
      request_id: 'request-1',
      username: 'current-user',
      global_model_name: 'seedance-public',
      model: 'seedance-public',
      mapped_model: 'provider-internal-model',
      observed_model: 'provider-secret-version',
      prompt: '一只奔跑的猫',
      status: 'completed',
      progress_percent: 100,
      provider_name: 'internal-provider',
      video_available: true,
      video_url: 'https://upstream.example.test/signed-secret-video',
      total_tokens: 10,
      input_tokens: 4,
      output_tokens: 6,
      cost: 0.25,
      actual_cost: 0.01,
      created_at: '2026-08-19T10:00:00Z',
    }],
    total: 1,
    page: 1,
    page_size: 20,
    pages: 1,
  })
  taskApiMocks.getStats.mockResolvedValue({
    total: 1,
    by_status: { completed: 1 },
    by_model: { 'seedance-public': 1 },
    today_count: 1,
    processing_count: 0,
  })
  taskApiMocks.getVideoBlob.mockResolvedValue(new Blob(['video'], { type: 'video/mp4' }))
})

afterEach(() => {
  for (const { app, root } of mountedApps.splice(0)) {
    app.unmount()
    root.remove()
  }
})

describe('VideoTasksPanel user scope', () => {
  it('loads playback through the authenticated blob API and revokes the object URL', async () => {
    const root = document.createElement('div')
    document.body.appendChild(root)
    const app = createApp({
      render: () => h(VideoTasksPanel, { active: true, scope: 'user' }),
    })
    app.mount(root)
    mountedApps.push({ app, root })

    await settle()

    expect(createVideoTasksApiMock).toHaveBeenCalledWith('user')
    expect(root.textContent).toContain('seedance-public')
    expect(root.textContent).not.toContain('provider-internal-model')
    expect(root.textContent).not.toContain('provider-secret-version')
    expect(root.textContent).toContain('$0.25')
    expect(root.textContent).not.toContain('$0.01')
    expect(root.querySelector('[title="查看计费明细"]')).toBeNull()
    expect(root.querySelector('video')).toBeNull()

    const previewButton = root.querySelector<HTMLButtonElement>('button[title="点击播放"]')
    expect(previewButton).not.toBeNull()
    previewButton?.click()
    await settle()

    expect(taskApiMocks.getVideoBlob).toHaveBeenCalledWith('task-1', expect.any(AbortSignal))
    expect(createObjectUrlMock).toHaveBeenCalledTimes(1)
    expect(root.querySelector('video')?.getAttribute('src')).toBe('blob:aether-video')
    expect(root.innerHTML).not.toContain('signed-secret-video')

    app.unmount()
    root.remove()
    mountedApps.splice(0)
    expect(revokeObjectUrlMock).toHaveBeenCalledWith('blob:aether-video')
  })

  it('aborts an in-flight protected media request when the panel unmounts', async () => {
    let requestSignal: AbortSignal | undefined
    taskApiMocks.getVideoBlob.mockImplementation((_taskId: string, signal?: AbortSignal) => {
      requestSignal = signal
      return new Promise<Blob>((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
      })
    })

    const root = document.createElement('div')
    document.body.appendChild(root)
    const app = createApp({
      render: () => h(VideoTasksPanel, { active: true, scope: 'user' }),
    })
    app.mount(root)
    mountedApps.push({ app, root })
    await settle()

    root.querySelector<HTMLButtonElement>('button[title="点击播放"]')?.click()
    await settle()
    expect(requestSignal?.aborted).toBe(false)

    app.unmount()
    root.remove()
    mountedApps.splice(0)
    await settle()

    expect(requestSignal?.aborted).toBe(true)
    expect(toastMock).not.toHaveBeenCalled()
    expect(createObjectUrlMock).not.toHaveBeenCalled()
  })
})
