import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createApp, nextTick, type App } from 'vue'

const { asyncApiMocks, authState } = vi.hoisted(() => ({
  asyncApiMocks: {
    list: vi.fn(),
    getStats: vi.fn(),
    getDetail: vi.fn(),
    cancel: vi.fn(),
  },
  authState: {
    canAccessAdmin: false,
    canOperateAdmin: false,
  },
}))

vi.mock('@/stores/auth', () => ({
  useAuthStore: () => authState,
}))

vi.mock('@/api/async-tasks', () => ({
  asyncTasksApi: asyncApiMocks,
}))

vi.mock('@/features/async-tasks/components/VideoTasksPanel.vue', async () => {
  const { defineComponent, h } = await import('vue')
  return {
    default: defineComponent({
      name: 'VideoTasksPanelStub',
      props: {
        active: { type: Boolean, required: true },
        scope: { type: String, required: true },
      },
      setup(props) {
        return () => h('div', {
          'data-testid': 'video-tasks-panel',
          'data-active': String(props.active),
          'data-scope': props.scope,
        })
      },
    }),
  }
})

vi.mock('@/features/usage/components', async () => {
  const { defineComponent, h } = await import('vue')
  return {
    RequestDetailDrawer: defineComponent({
      name: 'RequestDetailDrawerStub',
      setup: () => () => h('div', { 'data-testid': 'request-detail-drawer' }),
    }),
  }
})

import AsyncTasks from '@/views/admin/AsyncTasks.vue'

const mountedApps: Array<{ app: App, root: HTMLElement }> = []

async function settle() {
  for (let index = 0; index < 6; index += 1) {
    await Promise.resolve()
    await nextTick()
  }
}

beforeEach(() => {
  authState.canAccessAdmin = false
  authState.canOperateAdmin = false
  for (const mock of Object.values(asyncApiMocks)) mock.mockReset()
})

afterEach(() => {
  for (const { app, root } of mountedApps.splice(0)) {
    app.unmount()
    root.remove()
  }
})

describe('AsyncTasks ordinary user surface', () => {
  it('mounts only the user video panel and never calls admin system-task APIs', async () => {
    const root = document.createElement('div')
    document.body.appendChild(root)
    const app = createApp(AsyncTasks)
    app.mount(root)
    mountedApps.push({ app, root })

    await settle()

    const panel = root.querySelector('[data-testid="video-tasks-panel"]')
    expect(panel?.getAttribute('data-scope')).toBe('user')
    expect(panel?.getAttribute('data-active')).toBe('true')
    expect(root.querySelector('[data-testid="request-detail-drawer"]')).toBeNull()
    expect(asyncApiMocks.list).not.toHaveBeenCalled()
    expect(asyncApiMocks.getStats).not.toHaveBeenCalled()
    expect(asyncApiMocks.getDetail).not.toHaveBeenCalled()
    expect(asyncApiMocks.cancel).not.toHaveBeenCalled()
  })
})
