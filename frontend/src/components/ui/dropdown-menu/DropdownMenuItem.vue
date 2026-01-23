<script setup lang="ts">
import { DropdownMenuItem as DropdownMenuItemPrimitive } from 'radix-vue'
import { cn } from '@/lib/utils'
import { computed } from 'vue'

interface Props {
  class?: string
  disabled?: boolean
}

const props = defineProps<Props>()

defineEmits<{
  select: [event: Event]
}>()

const itemClass = computed(() =>
  cn(
    'relative flex cursor-pointer select-none items-center rounded-lg px-3 py-1.5 text-sm outline-none',
    'data-[highlighted]:bg-accent focus:bg-accent text-foreground',
    'transition-colors data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
    props.class
  )
)
</script>

<template>
  <DropdownMenuItemPrimitive
    :class="itemClass"
    :disabled="disabled"
    @select="$emit('select', $event)"
  >
    <slot />
  </DropdownMenuItemPrimitive>
</template>
