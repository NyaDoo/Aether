<template>
  <Card class="overflow-hidden">
    <!-- 表头 -->
    <div class="px-4 py-3 border-b border-border/60">
      <div class="flex items-center justify-between">
        <div class="flex items-baseline gap-2">
          <h4 class="text-sm font-semibold">映射规则</h4>
          <span class="text-xs text-muted-foreground">
            支持正则表达式 ({{ localMappings.length }}/{{ MAX_MAPPINGS_PER_MODEL }})
          </span>
        </div>
        <div class="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            class="h-7 w-7"
            title="添加规则"
            :disabled="localMappings.length >= MAX_MAPPINGS_PER_MODEL"
            @click="addMapping"
          >
            <Plus class="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            class="h-7 w-7"
            title="刷新"
            :disabled="props.loading"
            @click="$emit('refresh')"
          >
            <RefreshCw class="w-4 h-4" :class="{ 'animate-spin': props.loading }" />
          </Button>
        </div>
      </div>
    </div>

    <!-- 规则列表 -->
    <div v-if="localMappings.length > 0" class="divide-y">
      <div
        v-for="(mapping, index) in localMappings"
        :key="index"
      >
        <!-- 规则行 -->
        <div
          class="px-4 py-3 flex items-center gap-3 cursor-pointer hover:bg-muted/30 transition-colors"
          @click="toggleExpand(index)"
        >
          <ChevronRight
            class="w-4 h-4 text-muted-foreground transition-transform flex-shrink-0"
            :class="{ 'rotate-90': expandedIndex === index }"
          />
          <div class="flex-1 min-w-0">
            <Input
              v-model="localMappings[index]"
              placeholder="例如: claude-haiku-.*"
              :class="`font-mono text-sm ${mapping.trim() && !getMappingValidation(mapping).valid ? 'border-destructive' : ''}`"
              @click.stop
              @input="markDirty"
            />
            <!-- 验证错误提示 -->
            <div
              v-if="mapping.trim() && !getMappingValidation(mapping).valid"
              class="flex items-center gap-1 mt-1 text-xs text-destructive"
            >
              <AlertCircle class="w-3 h-3" />
              <span>{{ getMappingValidation(mapping).error }}</span>
            </div>
          </div>
          <!-- 匹配统计 -->
          <Badge
            v-if="getMappingValidation(mapping).valid && getMatchCount(mapping) > 0"
            variant="secondary"
            class="text-xs flex-shrink-0 h-6 leading-none"
          >
            {{ getMatchCount(mapping) }} 匹配
          </Badge>
          <Badge
            v-else-if="mapping.trim() && getMappingValidation(mapping).valid"
            variant="outline"
            class="text-xs text-muted-foreground flex-shrink-0 h-6 leading-none"
          >
            无匹配
          </Badge>
          <!-- 操作按钮 -->
          <div class="flex items-center gap-1 flex-shrink-0">
            <Button
              v-if="isDirty"
              variant="ghost"
              size="icon"
              class="h-7 w-7 text-muted-foreground hover:text-primary"
              title="保存"
              :disabled="saving || hasValidationErrors"
              @click.stop="saveMappings"
            >
              <Save v-if="!saving" class="w-4 h-4" />
              <RefreshCw v-else class="w-4 h-4 animate-spin" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              class="h-7 w-7 text-muted-foreground hover:text-destructive"
              title="删除"
              :disabled="saving"
              @click.stop="removeMapping(index)"
            >
              <Trash2 class="w-4 h-4" />
            </Button>
          </div>
        </div>

        <!-- 展开内容：匹配的 Key 列表 -->
        <div
          v-if="expandedIndex === index"
          class="border-t bg-muted/10 px-4 py-3"
        >
          <div v-if="loadingPreview" class="flex items-center justify-center py-4">
            <RefreshCw class="w-4 h-4 animate-spin text-muted-foreground" />
          </div>

          <div v-else-if="getMatchedKeysForMapping(mapping).length === 0" class="text-center py-4">
            <p class="text-sm text-muted-foreground">
              {{ mapping.trim() ? '此规则暂无匹配的 Key 白名单' : '请输入映射规则' }}
            </p>
          </div>

          <div v-else class="space-y-2">
            <div
              v-for="item in getMatchedKeysForMapping(mapping)"
              :key="item.keyId"
              class="bg-background rounded-md border p-3"
            >
              <div class="flex items-center gap-1.5 text-sm mb-2">
                <span class="text-muted-foreground">{{ item.providerName }}</span>
                <span class="text-muted-foreground">/</span>
                <span class="font-medium">{{ item.keyName }}</span>
                <span class="text-muted-foreground">·</span>
                <code class="text-xs text-muted-foreground/70">{{ item.maskedKey }}</code>
              </div>
              <div class="flex flex-wrap gap-1">
                <Badge
                  v-for="model in item.matchedModels"
                  :key="model"
                  variant="secondary"
                  class="text-xs font-mono"
                >
                  {{ model }}
                </Badge>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div
      v-else
      class="text-center py-32"
    >
      <GitMerge class="w-10 h-10 mx-auto text-muted-foreground/30 mb-3" />
      <p class="text-sm text-muted-foreground">
        暂无映射规则
      </p>
    </div>
  </Card>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted, computed } from 'vue'
import { Card, Button, Input, Badge } from '@/components/ui'
import { Plus, Trash2, GitMerge, RefreshCw, ChevronRight, Save, AlertCircle } from 'lucide-vue-next'
import { updateGlobalModel, getGlobalModel, getGlobalModelRoutingPreview } from '@/api/global-models'
import type { ModelRoutingPreviewResponse } from '@/api/endpoints/types'
import { log } from '@/utils/logger'
import { useToast } from '@/composables/useToast'

const props = defineProps<{
  globalModelId: string
  modelName: string
  mappings: string[]
  loading?: boolean
}>()
const emit = defineEmits<{
  update: [mappings: string[]]
  refresh: []
}>()
// 安全限制常量（与后端保持一致）
const MAX_MAPPINGS_PER_MODEL = 50
const MAX_MAPPING_LENGTH = 200

// 危险的正则模式（可能导致 ReDoS，与后端 model_permissions.py 保持一致）
// 注意：这些是用于检测用户输入字符串中的危险正则构造
const DANGEROUS_REGEX_PATTERNS = [
  /\([^)]*[+*]\)[+*]/,        // (x+)+, (x*)*, (x+)*, (x*)+
  /\([^)]*\)\{[0-9]+,\}/,     // (x){n,} 无上限
  /\(\.\*\)\{[0-9]+,\}/,      // (.*){n,} 贪婪量词 + 高重复
  /\(\.\+\)\{[0-9]+,\}/,      // (.+){n,} 贪婪量词 + 高重复
  /\([^)]*\|[^)]*\)[+*]/,     // (a|b)+ 选择分支 + 量词
  /\(\.\*\)\+/,               // (.*)+
  /\(\.\+\)\+/,               // (.+)+
  /\([^)]*\*\)[+*]/,          // 嵌套量词: (a*)+
  /\(\\w\+\)\+/,              // (\w+)+ - 检测字面量 \w
  /\(\.\*\)\*/,               // (.*)*
  /\(.*\+.*\)\+/,             // (a+b)+ 更通用的嵌套量词检测
  /\[.*\]\{[0-9]+,\}\{/,      // [x]{n,}{m,} 嵌套量词
  /\.{2,}\*/,                 // ..* 连续通配
  /\([^)]*\|[^)]*\)\*/,       // (a|a)* 选择分支 + 星号
  /\{[0-9]{2,},\}/,           // {10,} 高重复次数无上限
  /\(\[.*\]\+\)\+/,           // ([x]+)+ 字符类嵌套量词
  // 补充的危险模式（与后端保持一致）
  /\([^)]*[+*]\)\{[0-9]+,/,   // (a+){n,} 量词后跟大括号量词
  /\(\([^)]*[+*]\)[+*]\)/,    // ((a+)+) 三层嵌套量词
  /\(\?:[^)]*[+*]\)[+*]/,     // (?:a+)+ 非捕获组嵌套量词
]

// 正则匹配安全限制（与后端保持一致）
const REGEX_MATCH_MAX_INPUT_LENGTH = 200

const { success: toastSuccess, error: toastError } = useToast()

// 本地状态
const localMappings = ref<string[]>([...props.mappings])
const originalMappings = ref<string[]>([...props.mappings])  // 用于保存失败时恢复
const isDirty = ref(false)
const saving = ref(false)
const expandedIndex = ref<number | null>(null)

// 匹配预览状态
const loadingPreview = ref(false)
const routingData = ref<ModelRoutingPreviewResponse | null>(null)

// 正则编译缓存（简单的 LRU 实现）
const REGEX_CACHE_MAX_SIZE = 100

class LRURegexCache {
  private cache = new Map<string, RegExp | null>()
  private maxSize: number

  constructor(maxSize: number) {
    this.maxSize = maxSize
  }

  get(key: string): RegExp | null | undefined {
    if (!this.cache.has(key)) return undefined
    // 移到最后（LRU）
    const value = this.cache.get(key)!
    this.cache.delete(key)
    this.cache.set(key, value)
    return value
  }

  set(key: string, value: RegExp | null): void {
    // 如果已存在，先删除（会重新添加到最后）
    if (this.cache.has(key)) {
      this.cache.delete(key)
    } else if (this.cache.size >= this.maxSize) {
      // 缓存已满，删除最早的条目
      const firstKey = this.cache.keys().next().value as string | undefined
      if (firstKey !== undefined) {
        this.cache.delete(firstKey)
      }
    }
    this.cache.set(key, value)
  }

  clear(): void {
    this.cache.clear()
  }

  get size(): number {
    return this.cache.size
  }
}

const regexCache = new LRURegexCache(REGEX_CACHE_MAX_SIZE)

interface MatchedKeyForMapping {
  keyId: string
  keyName: string
  maskedKey: string
  providerName: string
  matchedModels: string[]
}

interface ValidationResult {
  valid: boolean
  error?: string
}

/**
 * 验证映射规则是否安全
 */
function validateMappingPattern(pattern: string): ValidationResult {
  if (!pattern || !pattern.trim()) {
    return { valid: false, error: '规则不能为空' }
  }

  if (pattern.length > MAX_MAPPING_LENGTH) {
    return { valid: false, error: `规则过长 (最大 ${MAX_MAPPING_LENGTH} 字符)` }
  }

  // 检查危险模式
  for (const dangerous of DANGEROUS_REGEX_PATTERNS) {
    if (dangerous.test(pattern)) {
      return { valid: false, error: '规则包含潜在危险的正则构造' }
    }
  }

  // 尝试编译验证语法
  try {
    new RegExp(`^${pattern}$`, 'i')
  } catch {
    return { valid: false, error: `正则表达式语法错误` }
  }

  return { valid: true }
}

/**
 * 获取映射的验证状态
 */
function getMappingValidation(mapping: string): ValidationResult {
  if (!mapping.trim()) {
    return { valid: true } // 空值暂不报错，保存时过滤
  }
  return validateMappingPattern(mapping)
}

/**
 * 检查是否有验证错误
 */
const hasValidationErrors = computed(() => {
  return localMappings.value.some(mapping => {
    if (!mapping.trim()) return false
    return !validateMappingPattern(mapping).valid
  })
})

/**
 * 安全的正则匹配（带缓存和保护）
 */
function matchPattern(pattern: string, text: string): boolean {
  // 快速路径：精确匹配
  if (pattern.toLowerCase() === text.toLowerCase()) {
    return true
  }

  // 长度检查
  if (pattern.length > MAX_MAPPING_LENGTH) {
    return false
  }

  // 危险模式检查
  for (const dangerous of DANGEROUS_REGEX_PATTERNS) {
    if (dangerous.test(pattern)) {
      return false
    }
  }

  // 使用 LRU 缓存
  let regex = regexCache.get(pattern)
  if (regex === undefined) {
    try {
      regex = new RegExp(`^${pattern}$`, 'i')
      regexCache.set(pattern, regex)
    } catch {
      regexCache.set(pattern, null)
      return false
    }
  }

  if (regex === null) {
    return false
  }

  try {
    // 额外保护：限制正则匹配的输入长度（与后端保持一致）
    const matchInput = text.slice(0, REGEX_MATCH_MAX_INPUT_LENGTH)
    return regex.test(matchInput)
  } catch {
    return false
  }
}

// 获取指定映射匹配的 Key 列表
function getMatchedKeysForMapping(mapping: string): MatchedKeyForMapping[] {
  if (!routingData.value || !mapping.trim()) return []

  // 使用 Map 按 keyId 去重并合并匹配结果
  const keyMap = new Map<string, MatchedKeyForMapping>()

  for (const provider of routingData.value.providers) {
    for (const endpoint of provider.endpoints) {
      for (const key of endpoint.keys) {
        if (!key.allowed_models || key.allowed_models.length === 0) continue

        const matchedModels: string[] = []
        for (const allowedModel of key.allowed_models) {
          if (matchPattern(mapping, allowedModel)) {
            matchedModels.push(allowedModel)
          }
        }

        if (matchedModels.length > 0) {
          const existing = keyMap.get(key.id)
          if (existing) {
            // 合并匹配结果（去重）
            const mergedModels = new Set([...existing.matchedModels, ...matchedModels])
            existing.matchedModels = Array.from(mergedModels)
          } else {
            keyMap.set(key.id, {
              keyId: key.id,
              keyName: key.name,
              maskedKey: key.masked_key,
              providerName: provider.name,
              matchedModels,
            })
          }
        }
      }
    }
  }

  return Array.from(keyMap.values())
}

// 获取指定映射的匹配数量
function getMatchCount(mapping: string): number {
  return getMatchedKeysForMapping(mapping).reduce((sum, item) => sum + item.matchedModels.length, 0)
}

function toggleExpand(index: number) {
  expandedIndex.value = expandedIndex.value === index ? null : index
}

watch(() => props.mappings, (newAliases) => {
  localMappings.value = [...newAliases]
  originalMappings.value = [...newAliases]
  isDirty.value = false
}, { deep: true })

// globalModelId 变化时清空缓存并重新加载预览
watch(() => props.globalModelId, () => {
  regexCache.clear()
  loadMatchPreview()
})

function markDirty() {
  isDirty.value = true
}

function addMapping() {
  if (localMappings.value.length >= MAX_MAPPINGS_PER_MODEL) {
    toastError(`最多支持 ${MAX_MAPPINGS_PER_MODEL} 条映射规则`)
    return
  }
  localMappings.value.push('')
  isDirty.value = true
  expandedIndex.value = localMappings.value.length - 1
}

async function removeMapping(index: number) {
  localMappings.value.splice(index, 1)
  if (expandedIndex.value === index) {
    expandedIndex.value = null
  } else if (expandedIndex.value !== null && expandedIndex.value > index) {
    expandedIndex.value--
  }
  // 删除后自动保存
  await saveMappings()
}

async function saveMappings() {
  const cleanedMappings = localMappings.value
    .map(a => a.trim())
    .filter(a => a.length > 0)

  saving.value = true
  try {
    const currentModel = await getGlobalModel(props.globalModelId)
    const currentConfig = currentModel.config || {}

    const updatedConfig = {
      ...currentConfig,
      model_mappings: cleanedMappings.length > 0 ? cleanedMappings : undefined,
    }

    if (!updatedConfig.model_mappings || updatedConfig.model_mappings.length === 0) {
      delete updatedConfig.model_mappings
    }

    await updateGlobalModel(props.globalModelId, {
      config: updatedConfig,
    })

    localMappings.value = cleanedMappings
    originalMappings.value = [...cleanedMappings]  // 更新原始值
    isDirty.value = false

    toastSuccess('映射规则已保存')
    emit('update', cleanedMappings)
  } catch (err) {
    log.error('保存映射规则失败:', err)
    toastError('保存失败，请重试')
    // 保存失败时恢复到原始值
    localMappings.value = [...originalMappings.value]
    isDirty.value = false
  } finally {
    saving.value = false
  }
}

async function loadMatchPreview() {
  // 清空正则缓存，确保使用最新数据
  regexCache.clear()
  loadingPreview.value = true
  try {
    routingData.value = await getGlobalModelRoutingPreview(props.globalModelId)
  } catch (err) {
    log.error('加载匹配预览失败:', err)
  } finally {
    loadingPreview.value = false
  }
}

onMounted(() => {
  loadMatchPreview()
})

// 组件卸载时清理缓存，防止内存泄漏
onUnmounted(() => {
  regexCache.clear()
})
</script>
