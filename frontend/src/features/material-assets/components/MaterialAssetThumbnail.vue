<template>
  <img
    v-if="objectUrl"
    :src="objectUrl"
    :alt="asset.name"
    class="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
    loading="lazy"
  >
  <Loader2
    v-else-if="loading"
    class="h-5 w-5 animate-spin text-primary"
    aria-label="正在加载素材缩略图"
  />
  <ImageOff
    v-else-if="failed"
    class="h-5 w-5 text-muted-foreground/60"
    aria-label="素材缩略图不可用"
  />
  <Image
    v-else
    class="h-5 w-5 text-muted-foreground/60"
  />
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'
import { Image, ImageOff, Loader2 } from 'lucide-vue-next'

import {
  createMaterialAssetsApi,
  type MaterialAsset,
  type MaterialAssetScope,
} from '@/api/material-assets'
import {
  materialAssetMediaType,
  normalizeMaterialAssetStatus,
} from '@/features/material-assets/utils/materialAssetPresentation'

const props = defineProps<{
  asset: MaterialAsset
  scope: MaterialAssetScope
  ownerUserId?: string
}>()

const api = createMaterialAssetsApi(props.scope)
const objectUrl = ref('')
const loading = ref(false)
const failed = ref(false)
let previewController: AbortController | null = null

function releaseObjectUrl() {
  if (!objectUrl.value) return
  URL.revokeObjectURL(objectUrl.value)
  objectUrl.value = ''
}

async function loadThumbnail() {
  previewController?.abort()
  previewController = null
  releaseObjectUrl()
  failed.value = false

  if (
    normalizeMaterialAssetStatus(props.asset.status) !== 'active'
    || materialAssetMediaType(props.asset) !== 'image'
  ) {
    loading.value = false
    return
  }

  const controller = new AbortController()
  previewController = controller
  loading.value = true
  try {
    const blob = await api.getPreviewBlob(
      props.asset.id,
      controller.signal,
      props.ownerUserId,
      props.asset.preview_url,
    )
    if (controller.signal.aborted) return
    objectUrl.value = URL.createObjectURL(blob)
  } catch {
    if (!controller.signal.aborted) failed.value = true
  } finally {
    if (previewController === controller) {
      previewController = null
      loading.value = false
    }
  }
}

watch(
  () => [props.asset.id, props.asset.status, props.asset.updated_at, props.asset.preview_url, props.ownerUserId],
  () => { void loadThumbnail() },
  { immediate: true },
)

onBeforeUnmount(() => {
  previewController?.abort()
  releaseObjectUrl()
})
</script>
