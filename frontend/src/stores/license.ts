import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { licenseApi, type LicenseStatus } from '@/api/license'
import { setLicenseDemoMode } from '@/config/demo'

export const useLicenseStore = defineStore('license', () => {
  const status = ref<LicenseStatus | null>(null)
  const loading = ref(false)
  const loaded = ref(false)
  const error = ref<string | null>(null)

  const isLicensed = computed(() => status.value?.licensed === true)
  const isDemoLocked = computed(() => status.value?.demo_mode !== false)

  function applyStatus(nextStatus: LicenseStatus): void {
    status.value = nextStatus
    loaded.value = true
    error.value = null
    setLicenseDemoMode(nextStatus.demo_mode, nextStatus.reason)
  }

  async function fetchStatus(force = false): Promise<LicenseStatus | null> {
    if (loaded.value && !force) return status.value
    loading.value = true
    try {
      const nextStatus = await licenseApi.getStatus()
      applyStatus(nextStatus)
      return nextStatus
    } catch (err) {
      error.value = err instanceof Error ? err.message : '许可证状态获取失败'
      setLicenseDemoMode(true, 'license_status_unavailable')
      return null
    } finally {
      loading.value = false
    }
  }

  async function activate(rawLicense: string): Promise<LicenseStatus> {
    loading.value = true
    try {
      const trimmed = rawLicense.trim()
      let payload: string | Record<string, unknown> = trimmed
      if (trimmed.startsWith('{')) {
        payload = JSON.parse(trimmed) as Record<string, unknown>
      }
      const nextStatus = await licenseApi.activate(payload)
      applyStatus(nextStatus)
      return nextStatus
    } finally {
      loading.value = false
    }
  }

  return {
    status,
    loading,
    loaded,
    error,
    isLicensed,
    isDemoLocked,
    fetchStatus,
    activate,
  }
})
