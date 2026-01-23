<script setup lang="ts">
import { SelectItem as SelectItemPrimitive, SelectItemIndicator, SelectItemText } from 'radix-vue'
import { Check } from 'lucide-vue-next'
import { cn } from '@/lib/utils'
import { computed } from 'vue'

interface Props {
  class?: string
  value: string
  disabled?: boolean
}

const props = defineProps<Props>()

const itemClass = computed(() =>
  cn(
    'relative flex w-full cursor-pointer select-none items-center rounded-lg py-1.5 pl-8 pr-2 text-sm outline-none',
    'data-[highlighted]:bg-accent focus:bg-accent text-foreground',
    'transition-colors data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
    props.class
  )
)
</script>

<template>
  <SelectItemPrimitive
    :class="itemClass"
    :value="value"
    :disabled="disabled"
  >
    <span class="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
      <SelectItemIndicator>
        <Check class="h-4 w-4" />
      </SelectItemIndicator>
    </span>
    <SelectItemText>
      <slot />
    </SelectItemText>
  </SelectItemPrimitive>
</template>
