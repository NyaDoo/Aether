<template>
  <DropdownMenu>
    <DropdownMenuTrigger as-child>
      <Button
        variant="ghost"
        size="icon"
        class="h-8 w-8 rounded-lg"
        :aria-label="`管理素材 ${asset.name}`"
        :title="`管理素材 ${asset.name}`"
      >
        <MoreHorizontal class="h-4 w-4" />
      </Button>
    </DropdownMenuTrigger>
    <DropdownMenuContent
      align="end"
      class="w-48"
    >
      <DropdownMenuItem
        :disabled="status !== 'active'"
        @select="emit('preview', asset)"
      >
        <Eye class="mr-2 h-4 w-4" />
        预览素材
      </DropdownMenuItem>
      <DropdownMenuItem @select="emit('copy', asset)">
        <Copy class="mr-2 h-4 w-4" />
        复制 asset://
      </DropdownMenuItem>
      <DropdownMenuItem
        :disabled="status !== 'active' || !supportsVideoReference"
        @select="emit('useForVideo', asset)"
      >
        <Video class="mr-2 h-4 w-4" />
        用于视频生成
      </DropdownMenuItem>
      <DropdownMenuItem
        :disabled="!canMutate"
        @select="emit('rename', asset)"
      >
        <Pencil class="mr-2 h-4 w-4" />
        重命名
      </DropdownMenuItem>
      <DropdownMenuItem
        class="text-destructive focus:text-destructive"
        :disabled="!canMutate || deleting"
        @select="emit('delete', asset)"
      >
        <Trash2 class="mr-2 h-4 w-4" />
        {{ deleting ? '删除中...' : '删除素材' }}
      </DropdownMenuItem>
    </DropdownMenuContent>
  </DropdownMenu>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Copy, Eye, MoreHorizontal, Pencil, Trash2, Video } from 'lucide-vue-next'

import type { MaterialAsset } from '@/api/material-assets'
import { Button } from '@/components/ui'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  materialAssetSupportsVideoReference,
  normalizeMaterialAssetStatus,
} from '@/features/material-assets/utils/materialAssetPresentation'

const props = defineProps<{
  asset: MaterialAsset
  canMutate: boolean
  deleting?: boolean
}>()

const emit = defineEmits<{
  preview: [asset: MaterialAsset]
  copy: [asset: MaterialAsset]
  useForVideo: [asset: MaterialAsset]
  rename: [asset: MaterialAsset]
  delete: [asset: MaterialAsset]
}>()

const status = computed(() => normalizeMaterialAssetStatus(props.asset.status))
const supportsVideoReference = computed(() => materialAssetSupportsVideoReference(props.asset))
</script>
