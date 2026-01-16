<template>
  <div class="rounded-lg border border-dashed border-border bg-card overflow-hidden">
    <!-- 标题栏 -->
    <div
      class="flex items-center gap-2 px-3 py-2 cursor-pointer hover:bg-muted/30 transition-colors"
      @click="$emit('toggle')"
    >
      <span
        class="text-xs font-medium shrink-0"
        :class="isLatest ? 'text-primary' : 'text-muted-foreground'"
      >
        #{{ turn.index }}
      </span>
      <TurnBadges :stats="turn.stats" />
      <!-- 折叠时显示摘要 -->
      <template v-if="!expanded">
        <span
          v-if="turn.summary.user"
          class="text-xs text-muted-foreground truncate"
        >
          {{ turn.summary.user }}
        </span>
        <span
          v-if="turn.summary.user && turn.summary.assistant"
          class="text-xs text-muted-foreground shrink-0"
        >→</span>
        <span
          v-if="turn.summary.assistant"
          class="text-xs text-muted-foreground truncate"
        >
          {{ turn.summary.assistant }}
        </span>
      </template>
      <component
        :is="expanded ? ChevronDown : ChevronRight"
        class="w-4 h-4 text-muted-foreground shrink-0 ml-auto"
      />
    </div>

    <!-- 展开内容 -->
    <div
      v-if="expanded"
      class="p-3 flex flex-col gap-2"
    >
      <!-- 用户消息 -->
      <div
        v-if="turn.user"
        class="rounded-lg bg-primary/[0.08] border border-primary/20 overflow-hidden"
      >
        <div class="flex items-center gap-2 px-3 py-2 text-xs font-medium text-primary">
          <User class="w-4 h-4" />
          <span>User</span>
        </div>
        <div class="px-3 pb-3 text-sm leading-relaxed">
          <BlockRenderer :blocks="userRenderBlocks" />
        </div>
      </div>

      <!-- 助手消息 -->
      <div
        v-if="turn.assistant"
        class="rounded-lg bg-muted/50 border border-border overflow-hidden"
      >
        <div class="flex items-center gap-2 px-3 py-2 text-xs font-medium text-foreground">
          <Bot class="w-4 h-4" />
          <span>Assistant</span>
          <TurnBadges
            :stats="turn.stats"
            :show-only="['thinking', 'tool']"
          />
        </div>
        <div class="px-3 pb-3 text-sm leading-relaxed">
          <BlockRenderer :blocks="assistantRenderBlocks" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { User, Bot, ChevronRight, ChevronDown } from 'lucide-vue-next'
import BlockRenderer from './BlockRenderer.vue'
import TurnBadges from './TurnBadges.vue'
import type { ConversationTurn } from '../../conversation/types'
import { contentBlocksToRenderBlocks } from '../../conversation/converter'

const props = defineProps<{
  turn: ConversationTurn
  expanded: boolean
  isLatest?: boolean
}>()

defineEmits<{
  toggle: []
}>()

const userRenderBlocks = computed(() => {
  if (!props.turn.user) return []
  return contentBlocksToRenderBlocks(props.turn.user.content)
})

const assistantRenderBlocks = computed(() => {
  if (!props.turn.assistant) return []
  return contentBlocksToRenderBlocks(props.turn.assistant.content)
})
</script>
