<template>
  <Dialog
    :model-value="isOpen"
    size="2xl"
    @update:model-value="handleDialogUpdate"
  >
    <template #header>
      <div class="border-b border-border px-6 py-4">
        <div class="flex items-center gap-3">
          <div class="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 flex-shrink-0">
            <Plus
              v-if="!isEditMode"
              class="h-5 w-5 text-primary"
            />
            <SquarePen
              v-else
              class="h-5 w-5 text-primary"
            />
          </div>
          <div class="flex-1 min-w-0">
            <h3 class="text-lg font-semibold text-foreground leading-tight">
              {{ isEditMode ? '编辑独立余额 API Key' : '创建独立余额 API Key' }}
            </h3>
            <p class="text-xs text-muted-foreground">
              {{ isEditMode ? '修改密钥名称、有效期和访问限制' : '用于非注册用户调用接口，不关联用户配额，必须设置余额限制' }}
            </p>
          </div>
        </div>
      </div>
    </template>

    <form @submit.prevent="handleSubmit">
      <div class="grid grid-cols-2 gap-0">
        <!-- 左侧：基础设置 -->
        <div class="pr-6 space-y-4">
          <div class="flex items-center gap-2 pb-2 border-b border-border/60">
            <Key class="h-4 w-4 text-muted-foreground" />
            <span class="text-sm font-medium">基础设置</span>
          </div>

          <div class="space-y-2">
            <Label
              for="form-name"
              class="text-sm font-medium"
            >密钥名称</Label>
            <Input
              id="form-name"
              v-model="form.name"
              type="text"
              placeholder="例如: 用户A专用"
              class="h-10"
            />
          </div>

          <!-- 初始余额 - 仅创建模式显示 -->
          <div
            v-if="!isEditMode"
            class="space-y-2"
          >
            <Label
              for="form-balance"
              class="text-sm font-medium"
            >初始余额 (USD) <span class="text-rose-500">*</span></Label>
            <Input
              id="form-balance"
              :model-value="form.initial_balance_usd ?? ''"
              type="number"
              step="0.01"
              min="0.01"
              required
              placeholder="10.00"
              class="h-10"
              @update:model-value="(v) => form.initial_balance_usd = parseNumberInput(v, { allowFloat: true }) ?? 10"
            />
            <p class="text-xs text-muted-foreground">
              独立Key必须设置余额限制，最小值 $0.01
            </p>
          </div>

          <div class="space-y-2">
            <Label
              for="form-expire-days"
              class="text-sm font-medium"
            >有效期设置</Label>
            <div class="flex items-center gap-2">
              <Input
                id="form-expire-days"
                :model-value="form.expire_days ?? ''"
                type="number"
                min="1"
                max="3650"
                placeholder="天数"
                :class="form.never_expire ? 'flex-1 h-9 opacity-50' : 'flex-1 h-9'"
                :disabled="form.never_expire"
                @update:model-value="(v) => form.expire_days = parseNumberInput(v, { min: 1, max: 3650 })"
              />
              <label class="flex items-center gap-1.5 border rounded-md px-2 py-1.5 bg-muted/50 cursor-pointer text-xs whitespace-nowrap">
                <input
                  v-model="form.never_expire"
                  type="checkbox"
                  class="h-3.5 w-3.5 rounded border-gray-300 cursor-pointer"
                  @change="onNeverExpireChange"
                >
                永不过期
              </label>
              <label
                class="flex items-center gap-1.5 border rounded-md px-2 py-1.5 bg-muted/50 cursor-pointer text-xs whitespace-nowrap"
                :class="form.never_expire ? 'opacity-50' : ''"
              >
                <input
                  v-model="form.auto_delete_on_expiry"
                  type="checkbox"
                  class="h-3.5 w-3.5 rounded border-gray-300 cursor-pointer"
                  :disabled="form.never_expire"
                >
                到期删除
              </label>
            </div>
            <p class="text-xs text-muted-foreground">
              不勾选"到期删除"则仅禁用
            </p>
          </div>

          <div class="space-y-2">
            <Label
              for="form-rate-limit"
              class="text-sm font-medium"
            >速率限制 (请求/分钟)</Label>
            <Input
              id="form-rate-limit"
              :model-value="form.rate_limit ?? ''"
              type="number"
              min="1"
              max="10000"
              placeholder="留空不限制"
              class="h-10"
              @update:model-value="(v) => form.rate_limit = parseNumberInput(v, { min: 1, max: 10000 })"
            />
          </div>
        </div>

        <!-- 右侧：访问限制 -->
        <div class="pl-6 space-y-4 border-l border-border">
          <div class="flex items-center gap-2 pb-2 border-b border-border/60">
            <Shield class="h-4 w-4 text-muted-foreground" />
            <span class="text-sm font-medium">访问限制</span>
            <span class="text-xs text-muted-foreground">(留空不限)</span>
          </div>

          <!-- Provider 多选下拉框 -->
          <div class="space-y-2">
            <Label class="text-sm font-medium">允许的 Provider</Label>
            <div class="relative">
              <button
                type="button"
                class="w-full h-10 px-3 border rounded-lg bg-background text-left flex items-center justify-between hover:bg-muted/50 transition-colors"
                @click="providerDropdownOpen = !providerDropdownOpen"
              >
                <span :class="form.allowed_providers.length ? 'text-foreground' : 'text-muted-foreground'">
                  {{ form.allowed_providers.length ? `已选择 ${form.allowed_providers.length} 个` : '全部可用' }}
                </span>
                <ChevronDown
                  class="h-4 w-4 text-muted-foreground transition-transform"
                  :class="providerDropdownOpen ? 'rotate-180' : ''"
                />
              </button>
              <div
                v-if="providerDropdownOpen"
                class="fixed inset-0 z-[80]"
                @click.stop="providerDropdownOpen = false"
              />
              <div
                v-if="providerDropdownOpen"
                class="absolute z-[90] w-full mt-1 bg-popover border rounded-lg shadow-lg max-h-48 overflow-y-auto"
              >
                <div
                  v-for="provider in providers"
                  :key="provider.id"
                  class="flex items-center gap-2 px-3 py-2 hover:bg-muted/50 cursor-pointer"
                  @click="toggleSelection('allowed_providers', provider.id)"
                >
                  <input
                    type="checkbox"
                    :checked="form.allowed_providers.includes(provider.id)"
                    class="h-4 w-4 rounded border-gray-300 cursor-pointer"
                    @click.stop
                    @change="toggleSelection('allowed_providers', provider.id)"
                  >
                  <span class="text-sm">{{ provider.display_name || provider.name }}</span>
                </div>
                <div
                  v-if="providers.length === 0"
                  class="px-3 py-2 text-sm text-muted-foreground"
                >
                  暂无可用 Provider
                </div>
              </div>
            </div>
          </div>

          <!-- API 格式多选下拉框 -->
          <div class="space-y-2">
            <Label class="text-sm font-medium">允许的 API 格式</Label>
            <div class="relative">
              <button
                type="button"
                class="w-full h-10 px-3 border rounded-lg bg-background text-left flex items-center justify-between hover:bg-muted/50 transition-colors"
                @click="apiFormatDropdownOpen = !apiFormatDropdownOpen"
              >
                <span :class="form.allowed_api_formats.length ? 'text-foreground' : 'text-muted-foreground'">
                  {{ form.allowed_api_formats.length ? `已选择 ${form.allowed_api_formats.length} 个` : '全部可用' }}
                </span>
                <ChevronDown
                  class="h-4 w-4 text-muted-foreground transition-transform"
                  :class="apiFormatDropdownOpen ? 'rotate-180' : ''"
                />
              </button>
              <div
                v-if="apiFormatDropdownOpen"
                class="fixed inset-0 z-[80]"
                @click.stop="apiFormatDropdownOpen = false"
              />
              <div
                v-if="apiFormatDropdownOpen"
                class="absolute z-[90] w-full mt-1 bg-popover border rounded-lg shadow-lg max-h-48 overflow-y-auto"
              >
                <div
                  v-for="format in allApiFormats"
                  :key="format"
                  class="flex items-center gap-2 px-3 py-2 hover:bg-muted/50 cursor-pointer"
                  @click="toggleSelection('allowed_api_formats', format)"
                >
                  <input
                    type="checkbox"
                    :checked="form.allowed_api_formats.includes(format)"
                    class="h-4 w-4 rounded border-gray-300 cursor-pointer"
                    @click.stop
                    @change="toggleSelection('allowed_api_formats', format)"
                  >
                  <span class="text-sm">{{ format }}</span>
                </div>
              </div>
            </div>
          </div>

          <!-- 模型多选下拉框 -->
          <ModelMultiSelect
            v-model="form.allowed_models"
            :models="globalModels"
          />
        </div>
      </div>
    </form>

    <template #footer>
      <Button
        variant="outline"
        type="button"
        class="h-10 px-5"
        @click="handleCancel"
      >
        取消
      </Button>
      <Button
        :disabled="saving"
        class="h-10 px-5"
        @click="handleSubmit"
      >
        {{ saving ? (isEditMode ? '更新中...' : '创建中...') : (isEditMode ? '更新' : '创建') }}
      </Button>
    </template>
  </Dialog>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import {
  Dialog,
  Button,
  Input,
  Label,
} from '@/components/ui'
import { Plus, SquarePen, Key, Shield, ChevronDown } from 'lucide-vue-next'
import { useFormDialog } from '@/composables/useFormDialog'
import { ModelMultiSelect } from '@/components/common'
import { getProvidersSummary } from '@/api/endpoints/providers'
import { getGlobalModels } from '@/api/global-models'
import { adminApi } from '@/api/admin'
import { log } from '@/utils/logger'
import { parseNumberInput } from '@/utils/form'
import type { ProviderWithEndpointsSummary, GlobalModelResponse } from '@/api/endpoints/types'

export interface StandaloneKeyFormData {
  id?: string
  name: string
  initial_balance_usd?: number
  expire_days?: number
  never_expire: boolean
  rate_limit?: number
  auto_delete_on_expiry: boolean
  allowed_providers: string[]
  allowed_api_formats: string[]
  allowed_models: string[]
}

const props = defineProps<{
  open: boolean
  apiKey: StandaloneKeyFormData | null
}>()

const emit = defineEmits<{
  close: []
  submit: [data: StandaloneKeyFormData]
}>()

const isOpen = computed(() => props.open)
const saving = ref(false)

// 下拉框状态
const providerDropdownOpen = ref(false)
const apiFormatDropdownOpen = ref(false)

// 选项数据
const providers = ref<ProviderWithEndpointsSummary[]>([])
const globalModels = ref<GlobalModelResponse[]>([])
const allApiFormats = ref<string[]>([])

// 表单数据
const form = ref<StandaloneKeyFormData>({
  name: '',
  initial_balance_usd: 10,
  expire_days: undefined,
  never_expire: true,
  rate_limit: undefined,
  auto_delete_on_expiry: false,
  allowed_providers: [],
  allowed_api_formats: [],
  allowed_models: []
})

function resetForm() {
  form.value = {
    name: '',
    initial_balance_usd: 10,
    expire_days: undefined,
    never_expire: true,
    rate_limit: undefined,
    auto_delete_on_expiry: false,
    allowed_providers: [],
    allowed_api_formats: [],
    allowed_models: []
  }
  providerDropdownOpen.value = false
  apiFormatDropdownOpen.value = false
}

function loadKeyData() {
  if (!props.apiKey) return
  form.value = {
    id: props.apiKey.id,
    name: props.apiKey.name || '',
    initial_balance_usd: props.apiKey.initial_balance_usd,
    expire_days: props.apiKey.expire_days,
    never_expire: props.apiKey.never_expire,
    rate_limit: props.apiKey.rate_limit,
    auto_delete_on_expiry: props.apiKey.auto_delete_on_expiry,
    allowed_providers: props.apiKey.allowed_providers || [],
    allowed_api_formats: props.apiKey.allowed_api_formats || [],
    allowed_models: props.apiKey.allowed_models || []
  }
}

const { isEditMode, handleDialogUpdate, handleCancel } = useFormDialog({
  isOpen: () => props.open,
  entity: () => props.apiKey,
  isLoading: saving,
  onClose: () => emit('close'),
  loadData: loadKeyData,
  resetForm,
})

// 加载选项数据
async function loadAccessRestrictionOptions() {
  try {
    const [providersData, modelsData, formatsData] = await Promise.all([
      getProvidersSummary(),
      getGlobalModels({ limit: 1000, is_active: true }),
      adminApi.getApiFormats()
    ])
    providers.value = providersData
    globalModels.value = modelsData.models || []
    allApiFormats.value = formatsData.formats?.map((f: any) => f.value) || []
  } catch (err) {
    log.error('加载访问限制选项失败:', err)
  }
}

// 切换选择
function toggleSelection(field: 'allowed_providers' | 'allowed_api_formats' | 'allowed_models', value: string) {
  const arr = form.value[field]
  const index = arr.indexOf(value)
  if (index === -1) {
    arr.push(value)
  } else {
    arr.splice(index, 1)
  }
}

// 永不过期切换
function onNeverExpireChange() {
  if (form.value.never_expire) {
    form.value.expire_days = undefined
    form.value.auto_delete_on_expiry = false
  }
}

// 提交表单
function handleSubmit() {
  emit('submit', { ...form.value })
}

// 设置保存状态
function setSaving(value: boolean) {
  saving.value = value
}

// 监听打开状态，加载选项数据
watch(isOpen, (val) => {
  if (val) {
    loadAccessRestrictionOptions()
  }
})

defineExpose({
  setSaving
})
</script>
