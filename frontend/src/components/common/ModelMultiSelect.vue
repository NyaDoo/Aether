<template>
  <div class="space-y-2">
    <Label class="text-sm font-medium">允许的模型</Label>
    <div class="relative">
      <button
        type="button"
        class="w-full h-10 px-3 border rounded-lg bg-background text-left flex items-center justify-between hover:bg-muted/50 transition-colors"
        @click="isOpen = !isOpen"
      >
        <span :class="modelValue.length ? 'text-foreground' : 'text-muted-foreground'">
          {{ modelValue.length ? `已选择 ${modelValue.length} 个` : '全部可用' }}
          <span
            v-if="invalidModels.length"
            class="text-destructive"
          >({{ invalidModels.length }} 个已失效)</span>
        </span>
        <ChevronDown
          class="h-4 w-4 text-muted-foreground transition-transform"
          :class="isOpen ? 'rotate-180' : ''"
        />
      </button>
      <div
        v-if="isOpen"
        class="fixed inset-0 z-[80]"
        @click.stop="isOpen = false"
      />
      <div
        v-if="isOpen"
        class="absolute z-[90] w-full mt-1 bg-popover border rounded-lg shadow-lg max-h-48 overflow-y-auto"
      >
        <!-- 失效模型（置顶显示，只能取消选择） -->
        <div
          v-for="modelName in invalidModels"
          :key="modelName"
          class="flex items-center gap-2 px-3 py-2 hover:bg-muted/50 cursor-pointer bg-destructive/5"
          @click="removeModel(modelName)"
        >
          <input
            type="checkbox"
            :checked="true"
            class="h-4 w-4 rounded border-gray-300 cursor-pointer"
            @click.stop
            @change="removeModel(modelName)"
          >
          <span class="text-sm text-destructive">{{ modelName }}</span>
          <span class="text-xs text-destructive/70">(已失效)</span>
        </div>
        <!-- 有效模型 -->
        <div
          v-for="model in models"
          :key="model.name"
          class="flex items-center gap-2 px-3 py-2 hover:bg-muted/50 cursor-pointer"
          @click="toggleModel(model.name)"
        >
          <input
            type="checkbox"
            :checked="modelValue.includes(model.name)"
            class="h-4 w-4 rounded border-gray-300 cursor-pointer"
            @click.stop
            @change="toggleModel(model.name)"
          >
          <span class="text-sm">{{ model.name }}</span>
        </div>
        <div
          v-if="models.length === 0 && invalidModels.length === 0"
          class="px-3 py-2 text-sm text-muted-foreground"
        >
          暂无可用模型
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { Label } from '@/components/ui'
import { ChevronDown } from 'lucide-vue-next'
import { useInvalidModels } from '@/composables/useInvalidModels'

export interface ModelWithName {
  name: string
}

const props = defineProps<{
  modelValue: string[]
  models: ModelWithName[]
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string[]]
}>()

const isOpen = ref(false)

// 检测失效模型
const { invalidModels } = useInvalidModels(
  computed(() => props.modelValue),
  computed(() => props.models)
)

function toggleModel(name: string) {
  const newValue = [...props.modelValue]
  const index = newValue.indexOf(name)
  if (index === -1) {
    newValue.push(name)
  } else {
    newValue.splice(index, 1)
  }
  emit('update:modelValue', newValue)
}

function removeModel(name: string) {
  const newValue = props.modelValue.filter(m => m !== name)
  emit('update:modelValue', newValue)
}
</script>
