<template>
  <Dialog
    :open="Boolean(asset)"
    size="6xl"
    :title="asset?.name || '素材预览'"
    :description="asset ? materialAssetUri(asset) : undefined"
    no-padding
    @update:open="handleOpenChange"
  >
    <div
      v-if="asset"
      class="grid min-h-[420px] lg:grid-cols-[minmax(0,1fr)_300px]"
    >
      <div class="flex min-h-[360px] items-center justify-center bg-black/90 p-4 lg:min-h-[560px]">
        <div
          v-if="previewLoading"
          class="text-center text-white/75"
        >
          <Loader2 class="mx-auto h-8 w-8 animate-spin text-white" />
          <p class="mt-3 text-sm">
            正在安全加载预览
          </p>
        </div>
        <div
          v-else-if="previewFailed || !objectUrl"
          class="max-w-sm text-center text-white/75"
        >
          <AlertCircle class="mx-auto mb-3 h-8 w-8 text-amber-300" />
          <p class="text-sm font-medium text-white">
            预览暂不可用
          </p>
          <p class="mt-1 text-xs leading-5">
            素材仍可通过素材 URI 在视频生成请求中使用。
          </p>
        </div>
        <img
          v-else-if="normalizedMediaType === 'image'"
          :src="objectUrl"
          :alt="asset.name"
          class="max-h-[72vh] max-w-full rounded-lg object-contain"
          @error="previewFailed = true"
        >
        <video
          v-else-if="normalizedMediaType === 'video'"
          :src="objectUrl"
          class="max-h-[72vh] max-w-full rounded-lg bg-black"
          controls
          playsinline
          preload="metadata"
          @error="previewFailed = true"
        />
        <audio
          v-else-if="normalizedMediaType === 'audio'"
          :src="objectUrl"
          class="w-full max-w-xl"
          controls
          preload="metadata"
          @error="previewFailed = true"
        />
        <div
          v-else
          class="text-center text-white/70"
        >
          <File class="mx-auto h-12 w-12" />
          <p class="mt-3 text-sm">
            该文件类型不支持内嵌预览
          </p>
        </div>
      </div>

      <aside class="space-y-5 border-l border-border/60 bg-background p-5">
        <div class="flex flex-wrap items-center gap-2">
          <Badge
            variant="outline"
            :class="statusClass"
          >
            {{ materialAssetStatusLabel(asset.status) }}
          </Badge>
          <Badge variant="secondary">
            {{ materialAssetMediaLabel(normalizedMediaType) }}
          </Badge>
          <Badge
            v-if="asset.requires_real_person_verification"
            variant="warning"
          >
            真人验证
          </Badge>
        </div>

        <dl class="space-y-3 text-sm">
          <div>
            <dt class="text-xs text-muted-foreground">
              素材 URI
            </dt>
            <dd class="mt-1 break-all font-mono text-xs text-foreground">
              {{ materialAssetUri(asset) }}
            </dd>
          </div>
          <div v-if="asset.group_name">
            <dt class="text-xs text-muted-foreground">
              素材组
            </dt>
            <dd class="mt-1">
              {{ asset.group_name }}
            </dd>
          </div>
          <div v-if="asset.mime_type">
            <dt class="text-xs text-muted-foreground">
              MIME 类型
            </dt>
            <dd class="mt-1 font-mono text-xs">
              {{ asset.mime_type }}
            </dd>
          </div>
          <div v-if="metadataSummary">
            <dt class="text-xs text-muted-foreground">
              媒体信息
            </dt>
            <dd class="mt-1">
              {{ metadataSummary }}
            </dd>
          </div>
          <div>
            <dt class="text-xs text-muted-foreground">
              创建时间
            </dt>
            <dd class="mt-1">
              {{ formatDate(asset.created_at) }}
            </dd>
          </div>
          <template v-if="showAdminMetadata">
            <div v-if="asset.username || asset.user_id">
              <dt class="text-xs text-muted-foreground">
                所属用户
              </dt>
              <dd class="mt-1">
                {{ asset.username || asset.user_id }}
              </dd>
            </div>
            <div v-if="asset.provider_name || asset.provider_id">
              <dt class="text-xs text-muted-foreground">
                Provider
              </dt>
              <dd class="mt-1">
                {{ asset.provider_name || asset.provider_id }}
              </dd>
            </div>
          </template>
        </dl>

        <div
          v-if="errorMessage"
          class="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-xs leading-5 text-destructive"
        >
          {{ errorMessage }}
        </div>
      </aside>
    </div>

    <template #footer>
      <Button
        variant="outline"
        @click="emit('close')"
      >
        关闭
      </Button>
      <Button
        v-if="asset"
        variant="outline"
        @click="emit('copy', asset)"
      >
        <Copy class="mr-2 h-4 w-4" />
        复制 asset://
      </Button>
      <Button
        v-if="asset && normalizeMaterialAssetStatus(asset.status) === 'active' && supportsVideoReference"
        @click="emit('useForVideo', asset)"
      >
        <Video class="mr-2 h-4 w-4" />
        用于视频生成
      </Button>
    </template>
  </Dialog>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { AlertCircle, Copy, File, Loader2, Video } from 'lucide-vue-next'

import {
  createMaterialAssetsApi,
  type MaterialAsset,
  type MaterialAssetScope,
} from '@/api/material-assets'
import { Button, Badge, Dialog } from '@/components/ui'
import { formatByteSize, formatDate } from '@/utils/format'
import {
  materialAssetErrorMessage,
  materialAssetMediaLabel,
  materialAssetMediaType,
  materialAssetStatusLabel,
  materialAssetSupportsVideoReference,
  materialAssetUri,
  normalizeMaterialAssetStatus,
} from '@/features/material-assets/utils/materialAssetPresentation'

const props = defineProps<{
  asset: MaterialAsset | null
  scope: MaterialAssetScope
  ownerUserId?: string
  showAdminMetadata?: boolean
}>()

const emit = defineEmits<{
  close: []
  copy: [asset: MaterialAsset]
  useForVideo: [asset: MaterialAsset]
}>()

const previewFailed = ref(false)
const previewLoading = ref(false)
const objectUrl = ref('')
const api = createMaterialAssetsApi(props.scope)
let previewController: AbortController | null = null

const normalizedMediaType = computed(() => props.asset ? materialAssetMediaType(props.asset) : 'unknown')
const supportsVideoReference = computed(() => props.asset
  ? materialAssetSupportsVideoReference(props.asset)
  : false)
const errorMessage = computed(() => props.asset ? materialAssetErrorMessage(props.asset) : null)
const statusClass = computed(() => {
  const status = normalizeMaterialAssetStatus(props.asset?.status)
  if (status === 'active') return 'border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
  if (status === 'failed') return 'border-destructive/30 bg-destructive/10 text-destructive'
  return 'border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400'
})
const metadataSummary = computed(() => {
  if (!props.asset) return ''
  const parts: string[] = []
  if (props.asset.width && props.asset.height) parts.push(`${props.asset.width}×${props.asset.height}`)
  if (props.asset.duration_seconds) parts.push(`${props.asset.duration_seconds}s`)
  if (props.asset.size_bytes !== null && props.asset.size_bytes !== undefined) {
    parts.push(formatByteSize(props.asset.size_bytes))
  }
  return parts.join(' · ')
})

function releaseObjectUrl() {
  if (!objectUrl.value) return
  URL.revokeObjectURL(objectUrl.value)
  objectUrl.value = ''
}

async function loadPreview() {
  previewController?.abort()
  previewController = null
  releaseObjectUrl()
  previewFailed.value = false

  if (!props.asset || normalizeMaterialAssetStatus(props.asset.status) !== 'active') {
    previewLoading.value = false
    return
  }

  const controller = new AbortController()
  previewController = controller
  previewLoading.value = true
  try {
    const blob = await api.getPreviewBlob(
      props.asset.id,
      controller.signal,
      props.ownerUserId,
    )
    if (controller.signal.aborted) return
    objectUrl.value = URL.createObjectURL(blob)
  } catch {
    if (!controller.signal.aborted) previewFailed.value = true
  } finally {
    if (previewController === controller) {
      previewController = null
      previewLoading.value = false
    }
  }
}

watch(
  () => [props.asset?.id, props.asset?.status, props.asset?.updated_at, props.ownerUserId],
  () => { void loadPreview() },
  { immediate: true },
)

onBeforeUnmount(() => {
  previewController?.abort()
  releaseObjectUrl()
})

function handleOpenChange(open: boolean) {
  if (!open) {
    previewController?.abort()
    releaseObjectUrl()
    emit('close')
  }
}
</script>
