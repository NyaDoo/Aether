<template>
  <div class="flex flex-wrap items-center gap-2">
    <Select
      v-model="selectedPreset"
    >
      <SelectTrigger
        class="h-8 w-32 text-xs border-border/60"
        :class="[presetTriggerClass]"
      >
        <SelectValue :placeholder="legacyT('选择时间段')" />
      </SelectTrigger>
      <SelectContent :searchable="false">
        <SelectItem
          v-for="preset in activePresetOptions"
          :key="preset"
          :value="preset"
        >
          {{ presetLabels[preset] }}
        </SelectItem>
      </SelectContent>
    </Select>

    <div
      v-if="selectedPreset === 'custom'"
      class="flex w-full min-w-0 flex-wrap items-center gap-2 sm:w-auto"
    >
      <Input
        v-model="startDate"
        :type="showTime ? 'datetime-local' : 'date'"
        :step="showTime ? 60 : undefined"
        :aria-label="showTime ? legacyT('开始时间') : legacyT('开始日期')"
        :title="showTime ? legacyT('开始时间（精确到分钟）') : undefined"
        :class="showTime ? 'h-8 w-36 min-w-0 max-w-full text-xs border-border/60 sm:w-44' : 'h-8 w-36 text-xs border-border/60'"
      />
      <span class="text-xs text-muted-foreground">{{ legacyT('至') }}</span>
      <Input
        v-model="endDate"
        :type="showTime ? 'datetime-local' : 'date'"
        :step="showTime ? 60 : undefined"
        :aria-label="showTime ? legacyT('结束时间') : legacyT('结束日期')"
        :title="showTime ? legacyT('结束时间（精确到分钟）') : undefined"
        :class="showTime ? 'h-8 w-36 min-w-0 max-w-full text-xs border-border/60 sm:w-44' : 'h-8 w-36 text-xs border-border/60'"
      />
    </div>

    <Select
      v-if="showGranularity"
      v-model="selectedGranularity"
    >
      <SelectTrigger class="h-8 w-24 text-xs border-border/60">
        <SelectValue :placeholder="legacyT('粒度')" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem
          v-if="allowHourly && canUseHourly"
          value="hour"
        >
          {{ legacyT('小时') }}
        </SelectItem>
        <SelectItem value="day">
          {{ legacyT('天') }}
        </SelectItem>
        <SelectItem value="week">
          {{ legacyT('周') }}
        </SelectItem>
        <SelectItem value="month">
          {{ legacyT('月') }}
        </SelectItem>
      </SelectContent>
    </Select>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Input
} from '@/components/ui'
import type { DateRangeParams } from '@/features/usage/types'
import { useI18n } from '@/i18n'

const props = withDefaults(defineProps<{
  modelValue: DateRangeParams
  showGranularity?: boolean
  allowHourly?: boolean
  /**
   * Show date and time controls for custom ranges. Values are emitted as
   * local `YYYY-MM-DDTHH:mm` strings so the selected timezone/offset can be
   * applied by the API layer. Preset ranges remain unchanged.
   */
  showTime?: boolean
  presetOptions?: SelectablePreset[]
  presetTriggerClass?: string
}>(), {
  presetOptions: () => ['today', 'yesterday', 'last7days', 'last30days', 'last90days', 'custom'],
  showTime: false,
  presetTriggerClass: undefined,
})
const emit = defineEmits<{
  'update:modelValue': [value: DateRangeParams]
}>()
const { legacyT } = useI18n()
const selectablePresets = ['today', 'yesterday', 'last7days', 'last30days', 'last90days', 'custom'] as const
type SelectablePreset = typeof selectablePresets[number]

const presetLabels = computed<Record<SelectablePreset, string>>(() => ({
  today: legacyT('今天'),
  yesterday: legacyT('昨天'),
  last7days: legacyT('最近7天'),
  last30days: legacyT('最近30天'),
  last90days: legacyT('最近90天'),
  custom: legacyT('自定义')
}))

const activePresetOptions = computed<SelectablePreset[]>(() => {
  const unique = new Set(props.presetOptions)
  const filtered = selectablePresets.filter((preset) => unique.has(preset))
  return filtered.length > 0 ? filtered : [...selectablePresets]
})

function defaultPreset(): SelectablePreset {
  const options = activePresetOptions.value
  if (options.includes('last7days')) return 'last7days'
  return options[0] ?? 'last7days'
}

function normalizePreset(value: DateRangeParams): SelectablePreset {
  if (value.preset && activePresetOptions.value.includes(value.preset as SelectablePreset)) {
    return value.preset as SelectablePreset
  }
  if (!value.preset && (value.start_date || value.end_date) && activePresetOptions.value.includes('custom')) {
    return 'custom'
  }
  return defaultPreset()
}

const selectedPreset = ref<SelectablePreset>(normalizePreset(props.modelValue))
const showTime = computed(() => props.showTime === true)

const DATE_ONLY_PATTERN = /^(\d{4}-\d{2}-\d{2})$/
const LOCAL_DATE_TIME_PATTERN = /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})(?::\d{2}(?:\.\d+)?)?$/

/**
 * Convert a range value into the format accepted by a datetime-local input.
 * Existing callers historically supplied date-only values; preserve their
 * inclusive-day meaning by using the beginning of the day for the lower
 * bound and the last minute of the day for the upper bound.
 */
function normalizeDateTimeInput(value: string | undefined, boundary: 'start' | 'end'): string {
  const trimmed = value?.trim() ?? ''
  if (!trimmed) return ''

  const dateOnly = DATE_ONLY_PATTERN.exec(trimmed)
  if (dateOnly) {
    return `${dateOnly[1]}T${boundary === 'start' ? '00:00' : '23:59'}`
  }

  const localDateTime = LOCAL_DATE_TIME_PATTERN.exec(trimmed)
  if (localDateTime) {
    return localDateTime[1]
  }

  // A value with an explicit timezone is not normally emitted by this
  // component, but can be supplied by an older/persisted caller. Convert it
  // to the browser's local clock before assigning it to datetime-local.
  const parsed = new Date(trimmed)
  if (!Number.isNaN(parsed.getTime())) {
    const year = String(parsed.getFullYear()).padStart(4, '0')
    const month = String(parsed.getMonth() + 1).padStart(2, '0')
    const day = String(parsed.getDate()).padStart(2, '0')
    const hours = String(parsed.getHours()).padStart(2, '0')
    const minutes = String(parsed.getMinutes()).padStart(2, '0')
    return `${year}-${month}-${day}T${hours}:${minutes}`
  }

  return trimmed
}

function normalizeDateInput(value: string | undefined): string {
  const trimmed = value?.trim() ?? ''
  if (!trimmed) return ''
  return trimmed.slice(0, 10)
}

function normalizeRangeInput(value: string | undefined, boundary: 'start' | 'end'): string {
  return showTime.value
    ? normalizeDateTimeInput(value, boundary)
    : normalizeDateInput(value)
}

const startDate = ref(normalizeRangeInput(props.modelValue.start_date, 'start'))
const endDate = ref(normalizeRangeInput(props.modelValue.end_date, 'end'))
const selectedGranularity = ref(props.modelValue.granularity || 'day')

const showGranularity = computed(() => props.showGranularity !== false)
const allowHourly = computed(() => props.allowHourly === true)

const canUseHourly = computed(() => {
  if (selectedPreset.value === 'today' || selectedPreset.value === 'yesterday') return true
  if (selectedPreset.value === 'custom' && startDate.value && endDate.value) {
    return startDate.value.slice(0, 10) === endDate.value.slice(0, 10)
  }
  return false
})

// 记录上次 emit 的值，避免重复触发
let lastEmittedValue: string | null = null

function buildEmitValue(): DateRangeParams {
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone
  const tz_offset_minutes = -new Date().getTimezoneOffset()

  if (selectedPreset.value === 'custom') {
    const start = startDate.value <= endDate.value ? startDate.value : endDate.value
    const end = endDate.value >= startDate.value ? endDate.value : startDate.value
    return {
      start_date: start,
      end_date: end,
      granularity: selectedGranularity.value,
      timezone,
      tz_offset_minutes
    }
  }

  return {
    preset: selectedPreset.value,
    granularity: selectedGranularity.value,
    timezone,
    tz_offset_minutes
  }
}

function getValueKey(value: DateRangeParams): string {
  // 只比较核心字段，忽略 timezone 和 tz_offset_minutes（这些每次都会重新计算）
  if (value.preset) {
    return `preset:${value.preset}:${value.granularity}`
  }
  return `custom:${value.start_date}:${value.end_date}:${value.granularity}`
}

watch(() => props.modelValue, (value) => {
  selectedPreset.value = normalizePreset(value)
  if (value.start_date !== undefined) startDate.value = normalizeRangeInput(value.start_date, 'start')
  if (value.end_date !== undefined) endDate.value = normalizeRangeInput(value.end_date, 'end')
  if (value.granularity) selectedGranularity.value = value.granularity
  // 同步更新 lastEmittedValue，避免外部设置值后触发重复 emit
  lastEmittedValue = getValueKey(value)
}, { deep: true })

// Keep the local input values valid if a caller toggles time precision at
// runtime.  Usage records enable it statically, while other consumers may
// reuse this picker with a reactive display preference.
watch(showTime, () => {
  startDate.value = normalizeRangeInput(startDate.value, 'start')
  endDate.value = normalizeRangeInput(endDate.value, 'end')
})

watch(activePresetOptions, () => {
  if (!activePresetOptions.value.includes(selectedPreset.value)) {
    selectedPreset.value = normalizePreset(props.modelValue)
  }
})

watch([selectedPreset, startDate, endDate, selectedGranularity], () => {
  if (!allowHourly.value || !canUseHourly.value) {
    if (selectedGranularity.value === 'hour') {
      selectedGranularity.value = 'day'
    }
  }

  if (selectedPreset.value === 'custom') {
    if (!startDate.value || !endDate.value) return
  }

  const newValue = buildEmitValue()
  const newKey = getValueKey(newValue)

  // 只有当值真正变化时才 emit，避免初始化时的重复触发
  if (newKey !== lastEmittedValue) {
    lastEmittedValue = newKey
    emit('update:modelValue', newValue)
  }
}, { immediate: true })
</script>
