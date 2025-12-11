<template>
  <Card class="p-4">
    <div class="flex items-center justify-between mb-3">
      <p class="text-sm font-semibold">{{ title }}</p>
      <div v-if="hasMultipleUsers && userLegend.length > 0" class="flex items-center gap-2 flex-wrap justify-end text-[11px]">
        <div
          v-for="user in userLegend"
          :key="user.id"
          class="flex items-center gap-1"
        >
          <div
            class="w-2.5 h-2.5 rounded-full"
            :style="{ backgroundColor: user.color }"
          />
          <span class="text-muted-foreground">{{ user.name }}</span>
        </div>
      </div>
    </div>
    <div v-if="loading" class="h-[160px] flex items-center justify-center">
      <div class="text-sm text-muted-foreground">Loading...</div>
    </div>
    <div v-else-if="hasData" class="h-[160px]">
      <ScatterChart :data="chartData" :options="chartOptions" />
    </div>
    <div v-else class="h-[160px] flex items-center justify-center text-sm text-muted-foreground">
      暂无请求间隔数据
    </div>
  </Card>
</template>

<script setup lang="ts">
import { computed, ref, onMounted, watch } from 'vue'
import Card from '@/components/ui/card.vue'
import ScatterChart from '@/components/charts/ScatterChart.vue'
import { cacheAnalysisApi, type IntervalTimelineResponse } from '@/api/cache'
import { meApi } from '@/api/me'
import type { ChartData, ChartOptions } from 'chart.js'

const props = withDefaults(defineProps<{
  title: string
  isAdmin: boolean
  hours?: number
}>(), {
  hours: 168  // 默认7天
})

const loading = ref(false)
const timelineData = ref<IntervalTimelineResponse | null>(null)
const primaryColor = ref('201, 100, 66')  // 默认主题色

// 获取主题色
function getPrimaryColor(): string {
  if (typeof window === 'undefined') return '201, 100, 66'
  // CSS 变量定义在 body 上，不是 documentElement
  const body = document.body
  const style = getComputedStyle(body)
  const rgb = style.getPropertyValue('--color-primary-rgb').trim()
  return rgb || '201, 100, 66'
}

onMounted(() => {
  primaryColor.value = getPrimaryColor()
  loadData()
})

// 预定义的颜色列表（用于区分不同用户）
const USER_COLORS = [
  'rgba(59, 130, 246, 0.7)',   // blue
  'rgba(236, 72, 153, 0.7)',   // pink
  'rgba(34, 197, 94, 0.7)',    // green
  'rgba(249, 115, 22, 0.7)',   // orange
  'rgba(168, 85, 247, 0.7)',   // purple
  'rgba(234, 179, 8, 0.7)',    // yellow
  'rgba(14, 165, 233, 0.7)',   // sky
  'rgba(239, 68, 68, 0.7)',    // red
  'rgba(20, 184, 166, 0.7)',   // teal
  'rgba(99, 102, 241, 0.7)',   // indigo
]

const hasData = computed(() =>
  timelineData.value && timelineData.value.points && timelineData.value.points.length > 0
)

const hasMultipleUsers = computed(() =>
  props.isAdmin && timelineData.value?.users && Object.keys(timelineData.value.users).length > 1
)

// 用户图例
const userLegend = computed(() => {
  if (!props.isAdmin || !timelineData.value?.users) return []

  const users = Object.entries(timelineData.value.users)
  return users.map(([userId, username], index) => ({
    id: userId,
    name: username || userId.slice(0, 8),
    color: USER_COLORS[index % USER_COLORS.length]
  }))
})

// 构建图表数据
const chartData = computed<ChartData<'scatter'>>(() => {
  if (!timelineData.value?.points) {
    return { datasets: [] }
  }

  const points = timelineData.value.points

  // 如果是管理员且有多个用户，按用户分组
  if (props.isAdmin && timelineData.value.users && Object.keys(timelineData.value.users).length > 1) {
    const userIds = Object.keys(timelineData.value.users)
    const userColorMap: Record<string, string> = {}
    userIds.forEach((userId, index) => {
      userColorMap[userId] = USER_COLORS[index % USER_COLORS.length]
    })

    // 按用户分组数据
    const groupedData: Record<string, Array<{ x: string; y: number }>> = {}
    for (const point of points) {
      const userId = point.user_id || 'unknown'
      if (!groupedData[userId]) {
        groupedData[userId] = []
      }
      groupedData[userId].push({ x: point.x, y: point.y })
    }

    // 创建每个用户的 dataset
    const datasets = Object.entries(groupedData).map(([userId, data]) => ({
      label: timelineData.value?.users?.[userId] || userId.slice(0, 8),
      data,
      backgroundColor: userColorMap[userId] || 'rgba(59, 130, 246, 0.6)',
      borderColor: userColorMap[userId] || 'rgba(59, 130, 246, 0.8)',
      pointRadius: 3,
      pointHoverRadius: 5,
    }))

    return { datasets }
  }

  // 单用户或用户视图：使用主题色
  return {
    datasets: [{
      label: '请求间隔',
      data: points.map(p => ({ x: p.x, y: p.y })),
      backgroundColor: `rgba(${primaryColor.value}, 0.6)`,
      borderColor: `rgba(${primaryColor.value}, 0.8)`,
      pointRadius: 3,
      pointHoverRadius: 5,
    }]
  }
})

const chartOptions = computed<ChartOptions<'scatter'>>(() => ({
  plugins: {
    legend: {
      display: false  // 使用自定义图例
    },
    tooltip: {
      callbacks: {
        label: (context) => {
          const point = context.raw as { x: string; y: number; _originalY?: number }
          const realY = point._originalY ?? point.y
          const datasetLabel = context.dataset.label || ''
          if (props.isAdmin && hasMultipleUsers.value) {
            return `${datasetLabel}: ${realY.toFixed(1)} 分钟`
          }
          return `间隔: ${realY.toFixed(1)} 分钟`
        }
      }
    }
  }
}))

async function loadData() {
  loading.value = true
  try {
    if (props.isAdmin) {
      // 管理员：获取所有用户数据
      timelineData.value = await cacheAnalysisApi.getIntervalTimeline({
        hours: props.hours,
        include_user_info: true,
        limit: 2000,
      })
    } else {
      // 普通用户：获取自己的数据
      timelineData.value = await meApi.getIntervalTimeline({
        hours: props.hours,
        limit: 1000,
      })
    }
  } catch (error) {
    console.error('加载请求间隔时间线失败:', error)
    timelineData.value = null
  } finally {
    loading.value = false
  }
}

watch(() => props.hours, () => {
  loadData()
})

watch(() => props.isAdmin, () => {
  loadData()
})
</script>
