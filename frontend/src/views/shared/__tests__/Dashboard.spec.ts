import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createApp, defineComponent, h, nextTick, type App } from 'vue'

import Dashboard from '../Dashboard.vue'

const dashboardApiMocks = vi.hoisted(() => ({
  getStats: vi.fn(),
  getDailyStats: vi.fn(),
  getRealtimeMetrics: vi.fn(),
}))

const authStoreState = vi.hoisted(() => ({
  canAccessAdmin: false,
  isAdmin: false,
  isAuditAdmin: false,
}))

vi.mock('@/stores/auth', () => ({
  useAuthStore: () => authStoreState,
}))

vi.mock('@/api/dashboard', () => ({
  dashboardApi: dashboardApiMocks,
}))

vi.mock('@/api/announcements', () => ({
  announcementApi: {
    getAnnouncements: vi.fn().mockResolvedValue({ items: [] }),
    markAsRead: vi.fn().mockResolvedValue({}),
  },
}))

vi.mock('@/components/charts/BarChart.vue', async () => {
  const { defineComponent, h } = await import('vue')
  return { default: defineComponent({ name: 'BarChartStub', setup: () => () => h('div') }) }
})

vi.mock('@/components/charts/DoughnutChart.vue', async () => {
  const { defineComponent, h } = await import('vue')
  return { default: defineComponent({ name: 'DoughnutChartStub', setup: () => () => h('div') }) }
})

vi.mock('@/components/charts/LineChart.vue', async () => {
  const { defineComponent, h } = await import('vue')
  return { default: defineComponent({ name: 'LineChartStub', setup: () => () => h('div') }) }
})

vi.mock('@/components/common', async () => {
  const { defineComponent, h } = await import('vue')
  return {
    TimeRangePicker: defineComponent({
      name: 'TimeRangePickerStub',
      setup() {
        return () => h('div')
      },
    }),
  }
})

vi.mock('@/components/ui', async () => {
  const { defineComponent, h } = await import('vue')
  const passthrough = (name: string, tag = 'div') => defineComponent({
    name,
    setup(_, { slots }) {
      return () => h(tag, slots.default?.())
    },
  })
  return {
    Card: passthrough('CardStub', 'section'),
    Badge: passthrough('BadgeStub', 'span'),
    Button: passthrough('ButtonStub', 'button'),
    Skeleton: defineComponent({ name: 'SkeletonStub', setup: () => () => h('div') }),
    Dialog: passthrough('DialogStub'),
    Table: passthrough('TableStub', 'table'),
    TableHeader: passthrough('TableHeaderStub', 'thead'),
    TableBody: passthrough('TableBodyStub', 'tbody'),
    TableRow: passthrough('TableRowStub', 'tr'),
    TableHead: passthrough('TableHeadStub', 'th'),
    TableCell: passthrough('TableCellStub', 'td'),
  }
})

vi.mock('lucide-vue-next', async () => {
  const Icon = defineComponent({
    name: 'IconStub',
    setup() {
      return () => h('span')
    },
  })
  return {
    Users: Icon,
    Activity: Icon,
    TrendingUp: Icon,
    DollarSign: Icon,
    Key: Icon,
    Hash: Icon,
    Zap: Icon,
    Bell: Icon,
    AlertCircle: Icon,
    AlertTriangle: Icon,
    Info: Icon,
    Wrench: Icon,
    Loader2: Icon,
    Clock: Icon,
    Database: Icon,
    Shuffle: Icon,
  }
})

const mountedApps: Array<{ app: App, root: HTMLElement }> = []

function mountDashboard() {
  const root = document.createElement('div')
  document.body.appendChild(root)
  const app = createApp(Dashboard)
  app.mount(root)
  mountedApps.push({ app, root })
  return root
}

async function settle() {
  for (let index = 0; index < 8; index += 1) {
    await Promise.resolve()
    await nextTick()
  }
}

beforeEach(() => {
  dashboardApiMocks.getStats.mockReset()
  dashboardApiMocks.getDailyStats.mockReset()
  dashboardApiMocks.getRealtimeMetrics.mockReset()
  authStoreState.canAccessAdmin = false
  authStoreState.isAdmin = false
  authStoreState.isAuditAdmin = false
  dashboardApiMocks.getDailyStats.mockResolvedValue({
    daily_stats: [],
    model_summary: [],
    period: { start_date: '2026-05-01', end_date: '2026-05-15', days: 15 },
  })
})

afterEach(() => {
  for (const { app, root } of mountedApps.splice(0)) {
    app.unmount()
    root.remove()
  }
  document.body.innerHTML = ''
})

describe('Dashboard ordinary user wallet card', () => {
  it('renders package and wallet balance split from mocked stats', async () => {
    dashboardApiMocks.getStats.mockResolvedValue({
      stats: [
        { name: 'API 密钥', value: '0', subValue: '活跃 0', icon: 'Activity' },
        { name: '本月请求', value: '0', subValue: '今日 0', icon: 'Users' },
        {
          name: '钱包余额',
          value: '$110.00',
          subValue: '套餐额度 $100.00 · 钱包余额 $10.00',
          icon: 'DollarSign',
        },
        { name: '本月 Token', value: '0', subValue: '输入 0 / 输出 0', icon: 'Zap' },
      ],
      today: { requests: 0, tokens: 0, cost: 0 },
      cache_stats: { cache_creation_tokens: 0, cache_read_tokens: 0, total_cache_tokens: 0 },
      token_breakdown: { input: 0, output: 0, cache_creation: 0, cache_read: 0 },
      monthly_cost: 0,
    })

    const root = mountDashboard()
    await settle()

    expect(root.textContent).toContain('$110.00')
    expect(root.textContent).toContain('套餐额度 $100.00 · 钱包余额 $10.00')
  })
})

describe('Dashboard refresh controls', () => {
  it('does not refresh the ordinary-user dashboard automatically', async () => {
    vi.useFakeTimers()
    dashboardApiMocks.getStats.mockResolvedValue({ stats: [] })

    try {
      const root = mountDashboard()
      await settle()

      expect(root.textContent).not.toContain('自动刷新')
      expect(dashboardApiMocks.getStats).toHaveBeenCalledTimes(1)
      expect(dashboardApiMocks.getDailyStats).toHaveBeenCalledTimes(1)

      await vi.advanceTimersByTimeAsync(60_000)
      await settle()

      expect(dashboardApiMocks.getStats).toHaveBeenCalledTimes(1)
      expect(dashboardApiMocks.getDailyStats).toHaveBeenCalledTimes(1)
    } finally {
      vi.useRealTimers()
    }
  })
})

describe('Dashboard realtime metrics', () => {
  const realtimeSnapshot = {
    rpm: 12.5,
    tpm: 3456,
    window_seconds: 60,
    as_of: '2026-08-27T12:34:56Z',
    semantics: {
      rpm: 'accepted_non_failed_requests',
      tpm: 'observed_token_deltas_including_failed',
      window: 'trailing_60_seconds',
      failed_requests: 'excluded_from_rpm_only',
    },
    storage_scope: 'shared',
  }

  function enableAdmin() {
    authStoreState.canAccessAdmin = true
    authStoreState.isAdmin = true
  }

  it('renders structured RPM/TPM values, window, timestamp, and semantics', async () => {
    enableAdmin()
    dashboardApiMocks.getStats.mockResolvedValue({
      stats: [
        { name: '今日请求 / 今日费用', value: '0 / $0.00', icon: 'Activity' },
        { name: '今日 Tokens', value: '0', icon: 'Hash' },
        { name: '全站 RPM / TPM', value: '0 / 0', subValue: '最近 60 秒', icon: 'Activity' },
        { name: '在线 / 启用用户', value: '0 / 0', icon: 'Users' },
      ],
    })
    dashboardApiMocks.getRealtimeMetrics.mockResolvedValue(realtimeSnapshot)

    const root = mountDashboard()
    await settle()

    expect(dashboardApiMocks.getRealtimeMetrics).toHaveBeenCalledTimes(1)
    expect(root.textContent).toContain('12.5 / 3.46K')
    expect(root.textContent).toContain('最近 60 秒')
    expect(root.textContent).toMatch(/截至 \d{2}:\d{2}:\d{2}/)
    expect(root.textContent).toContain('已接纳且未失败请求')
    expect(root.textContent).toContain('Token 增量')
    expect(root.textContent).toContain('全站共享')
  })

  it('loads realtime metrics without waiting for the aggregate dashboard', async () => {
    enableAdmin()
    let resolveStats!: (value: { stats: Array<{ name: string, value: string, icon: string }> }) => void
    dashboardApiMocks.getStats.mockReturnValue(new Promise((resolve) => {
      resolveStats = resolve
    }))
    dashboardApiMocks.getRealtimeMetrics.mockResolvedValue(realtimeSnapshot)

    const root = mountDashboard()
    await settle()

    expect(dashboardApiMocks.getRealtimeMetrics).toHaveBeenCalledTimes(1)

    resolveStats({ stats: [] })
    await settle()
    expect(root).toBeTruthy()
  })

  it('pauses polling while hidden and refreshes immediately when visible again', async () => {
    vi.useFakeTimers()
    enableAdmin()
    dashboardApiMocks.getStats.mockResolvedValue({ stats: [] })
    dashboardApiMocks.getRealtimeMetrics.mockResolvedValue(realtimeSnapshot)

    try {
      const root = mountDashboard()
      await settle()
      expect(root).toBeTruthy()
      expect(dashboardApiMocks.getRealtimeMetrics).toHaveBeenCalledTimes(1)

      Object.defineProperty(document, 'hidden', { configurable: true, value: true })
      document.dispatchEvent(new Event('visibilitychange'))
      await vi.advanceTimersByTimeAsync(15_000)
      await settle()
      expect(dashboardApiMocks.getRealtimeMetrics).toHaveBeenCalledTimes(1)

      Object.defineProperty(document, 'hidden', { configurable: true, value: false })
      document.dispatchEvent(new Event('visibilitychange'))
      await settle()
      expect(dashboardApiMocks.getRealtimeMetrics).toHaveBeenCalledTimes(2)

      await vi.advanceTimersByTimeAsync(5_000)
      await settle()
      expect(dashboardApiMocks.getRealtimeMetrics).toHaveBeenCalledTimes(3)
    } finally {
      Object.defineProperty(document, 'hidden', { configurable: true, value: false })
      vi.useRealTimers()
    }
  })
})
