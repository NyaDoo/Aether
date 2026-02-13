import { ref } from 'vue'
import apiClient from '@/api/client'

interface SiteInfo {
  site_name: string
  site_subtitle: string
}

// 模块级缓存，所有组件共享同一份数据
const siteName = ref('Aether')
const siteSubtitle = ref('AI Gateway')
const loaded = ref(false)
let fetchPromise: Promise<void> | null = null

async function fetchSiteInfo() {
  try {
    const response = await apiClient.get<SiteInfo>('/api/public/site-info')
    siteName.value = response.data.site_name
    siteSubtitle.value = response.data.site_subtitle
  } catch {
    // 加载失败时保持默认值
  } finally {
    loaded.value = true
  }
}

export function useSiteInfo() {
  if (!loaded.value && !fetchPromise) {
    fetchPromise = fetchSiteInfo()
  }
  return { siteName, siteSubtitle }
}
