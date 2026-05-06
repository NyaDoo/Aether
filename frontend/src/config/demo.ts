/**
 * Demo Mode Configuration
 * 用于静态托管环境的演示模式
 */

import { ref } from 'vue'

const forcedDemoMode = ref(false)
const forcedDemoReason = ref<string | null>(null)

// 检测是否为演示模式环境
export function isDemoMode(): boolean {
  const hostname = window.location.hostname
  return (
    forcedDemoMode.value ||
    hostname.includes('vercel.app') ||
    hostname.includes('netlify.app') ||
    hostname.includes('pages.dev') ||
    import.meta.env.VITE_DEMO_MODE === 'true'
  )
}

export function setLicenseDemoMode(enabled: boolean, reason: string | null = null): void {
  forcedDemoMode.value = enabled
  forcedDemoReason.value = reason
}

export function useDemoModeState() {
  return {
    forcedDemoMode,
    forcedDemoReason,
  }
}

// Demo 账号配置
export const DEMO_ACCOUNTS = {
  admin: {
    email: 'admin@demo.aether.io',
    password: 'demo123',
    hint: '管理员账号'
  },
  user: {
    email: 'user@demo.aether.io',
    password: 'demo123',
    hint: '普通用户'
  }
} as const

// Demo 模式提示信息
export const DEMO_MODE_INFO = {
  title: '演示模式',
  description: '当前处于演示模式，所有数据均为模拟数据，不会产生实际调用。',
  accountHint: '可使用以下演示账号登录：'
} as const
