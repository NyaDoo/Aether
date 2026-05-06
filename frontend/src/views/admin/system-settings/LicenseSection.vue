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
          <Label class="text-xs text-muted-foreground">版本</Label>
          <p class="mt-1 text-sm font-medium">
            {{ licenseStore.status?.edition || '-' }}
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
        v-if="licenseStore.status?.reason"
        class="rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-sm text-amber-700 dark:text-amber-300"
      >
        {{ reasonText }}
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
          粘贴签名后的许可证 JSON。激活成功后页面会刷新，并使用完整系统模式重新登录。
        </p>
      </div>

      <div class="flex items-center gap-3">
        <Button
          size="sm"
          :disabled="licenseStore.loading || !licenseInput.trim()"
          @click="handleActivate"
        >
          {{ licenseStore.loading ? '激活中...' : '激活许可证' }}
        </Button>
        <Button
          variant="outline"
          size="sm"
          :disabled="licenseStore.loading"
          @click="licenseStore.fetchStatus(true)"
        >
          刷新状态
        </Button>
      </div>
    </div>
  </CardSection>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
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

const statusLabel = computed(() => {
  const status = licenseStore.status
  if (!status) return '加载中'
  if (status.licensed) return '已授权'
  if (status.mode === 'expired') return '已过期'
  if (status.mode === 'invalid') return '无效'
  return '未授权'
})

const reasonText = computed(() => {
  const reason = licenseStore.status?.reason
  const map: Record<string, string> = {
    license_missing: '当前实例未激活许可证，已自动进入演示模式。',
    license_public_key_missing: '未配置许可证公钥，无法校验许可证。',
    license_signature_missing: '许可证缺少签名。',
    license_signature_invalid: '许可证签名无效。',
    license_expired: '许可证已过期。',
    instance_mismatch: '许可证绑定的实例与当前实例不匹配。',
    license_datetime_invalid: '许可证时间格式无效。',
  }
  return reason ? (map[reason] || reason) : ''
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

async function handleActivate() {
  try {
    await licenseStore.activate(licenseInput.value)
    showSuccess('许可证激活成功，正在刷新页面...')
    apiClient.clearAuth()
    setTimeout(() => window.location.reload(), 700)
  } catch (err) {
    showError(err instanceof Error ? err.message : '许可证激活失败')
  }
}
</script>
