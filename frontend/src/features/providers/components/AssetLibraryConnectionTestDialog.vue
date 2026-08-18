<template>
  <Dialog
    :open="open"
    :title="legacyT('素材库基础连通性测试')"
    :description="legacyT('使用已保存的密钥调用 ListAssetGroups，只验证基础连通性，不会创建或删除素材。')"
    :icon="Activity"
    size="sm"
    :persistent="testing"
    @update:open="$emit('update:open', $event)"
  >
    <div class="space-y-4">
      <div class="space-y-1">
        <p class="text-xs font-medium text-muted-foreground">
          {{ legacyT('测试密钥') }}
        </p>
        <p
          class="truncate text-sm font-medium text-foreground"
        >
          {{ keyName || legacyT('未命名密钥') }}
        </p>
      </div>

      <div class="space-y-1.5">
        <Label
          for="asset-library-test-endpoint"
          class="text-xs"
        >
          {{ legacyT('素材库端点') }}
        </Label>
        <Select
          :model-value="selectedEndpointId"
          :disabled="testing"
          @update:model-value="handleEndpointChange"
        >
          <SelectTrigger
            id="asset-library-test-endpoint"
            class="h-9"
          >
            <SelectValue :placeholder="legacyT('选择素材库端点')" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem
              v-for="endpoint in endpoints"
              :key="endpoint.id"
              :value="endpoint.id"
            >
              <span class="font-mono text-xs">{{ endpointDestination(endpoint) }}</span>
            </SelectItem>
          </SelectContent>
        </Select>
        <p class="text-xs leading-5 text-muted-foreground">
          {{ legacyT('测试将严格使用所选端点和当前密钥，不会自动切换。') }}
        </p>
      </div>
    </div>

    <template #footer>
      <Button
        variant="outline"
        :disabled="testing"
        @click="$emit('update:open', false)"
      >
        {{ legacyT('取消') }}
      </Button>
      <Button
        :disabled="testing || !selectedEndpointExists"
        @click="$emit('test')"
      >
        <Loader2
          v-if="testing"
          class="mr-2 h-4 w-4 animate-spin"
        />
        <Activity
          v-else
          class="mr-2 h-4 w-4"
        />
        {{ legacyT(testing ? '正在测试' : '开始测试') }}
      </Button>
    </template>
  </Dialog>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Activity, Loader2 } from 'lucide-vue-next'
import {
  Button,
  Dialog,
  Label,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui'
import type { ProviderEndpoint } from '@/api/endpoints'
import { useI18n } from '@/i18n'

const props = defineProps<{
  open: boolean
  endpoints: ProviderEndpoint[]
  keyName?: string | null
  selectedEndpointId: string
  testing?: boolean
}>()

const emit = defineEmits<{
  (event: 'update:open', value: boolean): void
  (event: 'update:selectedEndpointId', value: string): void
  (event: 'test'): void
}>()

const { legacyT } = useI18n()
const selectedEndpointExists = computed(() => props.endpoints.some(
  endpoint => endpoint.id === props.selectedEndpointId,
))

function handleEndpointChange(value: unknown): void {
  if (typeof value === 'string') {
    emit('update:selectedEndpointId', value)
  }
}

function endpointDestination(endpoint: ProviderEndpoint): string {
  try {
    const parsed = new URL(endpoint.base_url)
    const customPath = endpoint.custom_path?.trim().split(/[?#]/, 1)[0]
    const effectivePath = customPath || parsed.pathname || '/'
    const normalizedPath = effectivePath.startsWith('/') ? effectivePath : `/${effectivePath}`
    const shortId = endpoint.id.slice(0, 8)
    return `${parsed.protocol}//${parsed.host}${normalizedPath} · ${shortId}`
  } catch {
    return `${endpoint.api_format} · ${endpoint.id.slice(0, 8)}`
  }
}
</script>
