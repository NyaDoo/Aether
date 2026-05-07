<template>
  <CardSection
    title="许可证授权"
    description="管理当前实例的授权状态；未授权时系统仅允许演示模式"
  >
    <div class="space-y-5">
      <div class="grid gap-4 md:grid-cols-3">
        <div>
          <Label class="text-xs text-muted-foreground">授权状态</Label>
          <p class="mt-1 text-sm font-medium">
            {{ statusLabel }}
          </p>
        </div>
        <div>
          <Label class="text-xs text-muted-foreground">授权范围</Label>
          <p class="mt-1 text-sm font-medium">
            {{ licenseStore.status?.licensed ? '全部功能' : '-' }}
          </p>
        </div>
        <div>
          <Label class="text-xs text-muted-foreground">到期时间</Label>
          <p class="mt-1 text-sm font-medium">
            {{ formattedExpiresAt }}
          </p>
        </div>
      </div>

      <div
        v-if="licenseStore.status"
        class="rounded-lg border px-3 py-2 text-sm"
        :class="licenseStore.status.licensed
          ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
          : 'border-amber-500/20 bg-amber-500/10 text-amber-700 dark:text-amber-300'"
      >
        {{ licenseStore.status.licensed ? '许可证状态有效。' : '许可证状态无效。' }}
      </div>

      <div class="grid gap-4 md:grid-cols-2">
        <div>
          <Label class="text-xs text-muted-foreground">机器指纹</Label>
          <button
            type="button"
            class="mt-2 block max-w-full break-all text-left font-mono text-xs text-foreground underline-offset-4 hover:text-primary hover:underline disabled:pointer-events-none disabled:opacity-60"
            :disabled="machineFingerprint === '-'"
            @click="handleCopyMachineFingerprint"
          >
            {{ machineFingerprint }}
          </button>
        </div>
        <div>
          <Label class="text-xs text-muted-foreground">当前时间</Label>
          <p class="mt-1 text-sm font-medium">
            {{ currentTimeLabel }}
          </p>
        </div>
      </div>

      <div>
        <Label
          for="license-input"
          class="block text-sm font-medium"
        >
          许可证内容
        </Label>
        <Textarea
          id="license-input"
          v-model="licenseInput"
          class="mt-2 min-h-40 font-mono text-xs"
          placeholder="{ &quot;license_id&quot;: &quot;lic_xxx&quot;, ... }"
        />
        <p class="mt-2 text-xs text-muted-foreground">
          粘贴签名后的许可证 JSON。
        </p>
      </div>

      <div class="flex flex-wrap items-center gap-3">
        <Button
          size="sm"
          :disabled="licenseStore.loading || !licenseInput.trim()"
          @click="handleActivate"
        >
          {{ licenseStore.loading ? '激活中...' : '激活 / 修改许可证' }}
        </Button>
        <Button
          variant="outline"
          size="sm"
          :disabled="licenseStore.loading"
          @click="licenseStore.fetchStatus(true)"
        >
          刷新状态
        </Button>
        <Button
          variant="destructive"
          size="sm"
          :disabled="licenseStore.loading || !licenseStore.status?.can_deactivate"
          @click="handleDeactivate"
        >
          取消关联许可证
        </Button>
      </div>
    </div>
  </CardSection>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import Button from '@/components/ui/button.vue'
import Label from '@/components/ui/label.vue'
import Textarea from '@/components/ui/textarea.vue'
import { CardSection } from '@/components/layout'
import { useToast } from '@/composables/useToast'
import { useLicenseStore } from '@/stores/license'
import apiClient from '@/api/client'

const licenseStore = useLicenseStore()
const { success: showSuccess, error: showError } = useToast()
const licenseInput = ref('')
const currentTime = ref(new Date())
let currentTimeTimer: number | undefined

onMounted(() => {
  void licenseStore.fetchMachineBinding()
  currentTimeTimer = window.setInterval(() => {
    currentTime.value = new Date()
  }, 1000)
})

onUnmounted(() => {
  if (currentTimeTimer !== undefined) {
    window.clearInterval(currentTimeTimer)
  }
})

const statusLabel = computed(() => {
  const status = licenseStore.status
  if (!status) return '加载中'
  return status.licensed ? '有效' : '无效'
})

const formattedExpiresAt = computed(() => {
  const value = licenseStore.status?.expires_at
  if (!value) return '永久'
  try {
    return new Date(value).toLocaleDateString('zh-CN')
  } catch {
    return value
  }
})

const machineFingerprint = computed(() => {
  return (
    licenseStore.status?.machine_fingerprint ||
    licenseStore.machineBinding?.fingerprint ||
    '-'
  )
})

const currentTimeLabel = computed(() => currentTime.value.toLocaleString('zh-CN'))

async function handleCopyMachineFingerprint() {
  try {
    await navigator.clipboard.writeText(machineFingerprint.value)
    showSuccess('机器指纹已复制')
  } catch {
    showError('复制机器指纹失败')
  }
}

async function handleActivate() {
  try {
    await licenseStore.activate(licenseInput.value)
    showSuccess('许可证激活成功，正在刷新页面...')
    apiClient.clearAuth()
    setTimeout(() => window.location.reload(), 700)
  } catch {
    showError('许可证激活失败')
  }
}

async function handleDeactivate() {
  try {
    await licenseStore.deactivate()
    showSuccess('许可证关联已取消，正在刷新页面...')
    apiClient.clearAuth()
    setTimeout(() => window.location.reload(), 700)
  } catch {
    showError('取消关联失败')
  }
}
</script>
