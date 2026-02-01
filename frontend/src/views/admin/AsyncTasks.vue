<template>
  <div class="space-y-6 pb-8">
    <!-- 统计卡片 -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <Card variant="default" class="p-4">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
            <Zap class="w-5 h-5 text-primary" />
          </div>
          <div>
            <p class="text-2xl font-bold">{{ stats?.total ?? '-' }}</p>
            <p class="text-xs text-muted-foreground">总任务数</p>
          </div>
        </div>
      </Card>
      <Card variant="default" class="p-4">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
            <Loader2 class="w-5 h-5 text-blue-500" :class="{ 'animate-spin': (stats?.processing_count ?? 0) > 0 }" />
          </div>
          <div>
            <p class="text-2xl font-bold">{{ stats?.processing_count ?? stats?.by_status?.processing ?? '-' }}</p>
            <p class="text-xs text-muted-foreground">处理中</p>
          </div>
        </div>
      </Card>
      <Card variant="default" class="p-4">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center">
            <CheckCircle class="w-5 h-5 text-green-500" />
          </div>
          <div>
            <p class="text-2xl font-bold">{{ stats?.by_status?.completed ?? '-' }}</p>
            <p class="text-xs text-muted-foreground">已完成</p>
          </div>
        </div>
      </Card>
      <Card variant="default" class="p-4">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center">
            <Calendar class="w-5 h-5 text-amber-500" />
          </div>
          <div>
            <p class="text-2xl font-bold">{{ stats?.today_count ?? '-' }}</p>
            <p class="text-xs text-muted-foreground">今日任务</p>
          </div>
        </div>
      </Card>
    </div>

    <!-- 任务表格 -->
    <Card variant="default" class="overflow-hidden">
      <!-- 标题和筛选器 -->
      <div class="px-4 sm:px-6 py-3.5 border-b border-border/60">
        <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <h3 class="text-base font-semibold">异步任务</h3>
          <div class="flex items-center gap-2">
            <!-- 状态筛选 -->
            <Select v-model="filterStatus">
              <SelectTrigger class="w-28 h-8 text-xs border-border/60">
                <SelectValue placeholder="状态" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">全部状态</SelectItem>
                <SelectItem value="submitted">已提交</SelectItem>
                <SelectItem value="processing">处理中</SelectItem>
                <SelectItem value="completed">已完成</SelectItem>
                <SelectItem value="failed">失败</SelectItem>
                <SelectItem value="cancelled">已取消</SelectItem>
              </SelectContent>
            </Select>
            <!-- 模型筛选 -->
            <Input
              v-model="filterModel"
              type="text"
              placeholder="模型..."
              class="w-32 h-8 text-xs"
            />
            <!-- 刷新按钮 -->
            <Button
              variant="ghost"
              size="icon"
              class="h-8 w-8"
              :disabled="loading"
              @click="fetchTasks"
            >
              <RefreshCw class="w-3.5 h-3.5" :class="{ 'animate-spin': loading }" />
            </Button>
          </div>
        </div>
      </div>

      <!-- 加载状态 -->
      <div v-if="loading && !tasks.length" class="p-8 text-center">
        <Loader2 class="w-8 h-8 animate-spin mx-auto text-muted-foreground" />
        <p class="mt-2 text-sm text-muted-foreground">加载中...</p>
      </div>

      <!-- 空状态 -->
      <div v-else-if="!tasks.length" class="p-8 text-center">
        <Zap class="w-12 h-12 mx-auto text-muted-foreground/50" />
        <p class="mt-2 text-sm text-muted-foreground">暂无异步任务</p>
      </div>

      <!-- 任务列表 -->
      <div v-else class="divide-y divide-border/60">
        <div
          v-for="task in tasks"
          :key="task.id"
          class="px-4 sm:px-6 py-4 hover:bg-muted/30 cursor-pointer transition-colors"
          @click="openTaskDetail(task)"
        >
          <div class="flex items-start justify-between gap-4">
            <div class="flex-1 min-w-0">
              <!-- 模型和状态 -->
              <div class="flex items-center gap-2 mb-1">
                <Video v-if="isVideoTask(task)" class="w-4 h-4 text-muted-foreground" />
                <span class="font-medium text-sm">{{ task.model }}</span>
                <Badge :variant="getStatusVariant(task.status)">
                  {{ getStatusLabel(task.status) }}
                </Badge>
                <span v-if="task.progress_percent > 0 && task.status === 'processing'" class="text-xs text-muted-foreground">
                  {{ task.progress_percent }}%
                </span>
              </div>
              <!-- Prompt 摘要 -->
              <p class="text-sm text-muted-foreground truncate">{{ task.prompt }}</p>
              <!-- 元信息 -->
              <div class="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                <span class="flex items-center gap-1">
                  <User class="w-3 h-3" />
                  {{ task.username }}
                </span>
                <span class="flex items-center gap-1">
                  <Server class="w-3 h-3" />
                  {{ task.provider_name }}
                </span>
                <span class="flex items-center gap-1">
                  <Clock class="w-3 h-3" />
                  {{ formatDate(task.created_at) }}
                </span>
                <span v-if="task.duration_seconds" class="flex items-center gap-1">
                  <Timer class="w-3 h-3" />
                  {{ task.duration_seconds }}s
                </span>
              </div>
            </div>
            <!-- 操作 -->
            <div class="flex items-center gap-2">
              <Button
                v-if="canCancel(task.status)"
                variant="ghost"
                size="sm"
                class="text-red-500 hover:text-red-600 hover:bg-red-50"
                @click.stop="cancelTask(task)"
              >
                <XCircle class="w-4 h-4" />
              </Button>
              <ChevronRight class="w-4 h-4 text-muted-foreground" />
            </div>
          </div>
        </div>
      </div>

      <!-- 分页 -->
      <div v-if="totalPages > 1" class="px-4 sm:px-6 py-3 border-t border-border/60 flex items-center justify-between">
        <p class="text-xs text-muted-foreground">
          共 {{ total }} 条，第 {{ currentPage }}/{{ totalPages }} 页
        </p>
        <div class="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            :disabled="currentPage <= 1"
            @click="goToPage(currentPage - 1)"
          >
            上一页
          </Button>
          <Button
            variant="outline"
            size="sm"
            :disabled="currentPage >= totalPages"
            @click="goToPage(currentPage + 1)"
          >
            下一页
          </Button>
        </div>
      </div>
    </Card>

    <!-- 任务详情抽屉 -->
    <Teleport to="body">
      <Transition name="drawer">
        <div
          v-if="showDetail && selectedTask"
          class="fixed inset-0 z-50 flex justify-end"
          @click.self="showDetail = false"
        >
          <!-- 背景遮罩 -->
          <div
            class="absolute inset-0 bg-black/30 backdrop-blur-sm"
            @click="showDetail = false"
          />
          <!-- 抽屉内容 -->
          <Card class="relative h-full w-full sm:w-[600px] sm:max-w-[90vw] rounded-none shadow-2xl overflow-y-auto">
            <!-- 标题栏 -->
            <div class="sticky top-0 z-10 bg-background border-b p-4 sm:p-6">
              <div class="flex items-center justify-between">
                <h3 class="text-lg font-semibold">任务详情</h3>
                <Button variant="ghost" size="icon" @click="showDetail = false">
                  <X class="w-4 h-4" />
                </Button>
              </div>
            </div>
            <!-- 内容 -->
            <div class="p-4 sm:p-6 space-y-6">
          <!-- 状态和进度 -->
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-sm font-medium">状态</span>
              <Badge :variant="getStatusVariant(selectedTask.status)">
                {{ getStatusLabel(selectedTask.status) }}
              </Badge>
            </div>
            <div v-if="selectedTask.progress_percent > 0" class="space-y-1">
              <div class="flex justify-between text-xs text-muted-foreground">
                <span>进度</span>
                <span>{{ selectedTask.progress_percent }}%</span>
              </div>
              <div class="h-2 bg-muted rounded-full overflow-hidden">
                <div
                  class="h-full bg-primary transition-all"
                  :style="{ width: `${selectedTask.progress_percent}%` }"
                />
              </div>
            </div>
            <p v-if="selectedTask.progress_message" class="text-xs text-muted-foreground">
              {{ selectedTask.progress_message }}
            </p>
          </div>

          <!-- 错误信息 -->
          <div v-if="selectedTask.error_message" class="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
            <p class="text-sm text-red-600 dark:text-red-400">
              <span v-if="selectedTask.error_code" class="font-medium">[{{ selectedTask.error_code }}]</span>
              {{ selectedTask.error_message }}
            </p>
          </div>

          <!-- 基本信息 -->
          <div class="space-y-3">
            <h4 class="text-sm font-medium">基本信息</h4>
            <div class="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span class="text-muted-foreground">模型</span>
                <p class="font-medium">{{ selectedTask.model }}</p>
              </div>
              <div>
                <span class="text-muted-foreground">时长</span>
                <p class="font-medium">{{ selectedTask.duration_seconds }}s</p>
              </div>
              <div>
                <span class="text-muted-foreground">分辨率</span>
                <p class="font-medium">{{ selectedTask.resolution }}</p>
              </div>
              <div>
                <span class="text-muted-foreground">宽高比</span>
                <p class="font-medium">{{ selectedTask.aspect_ratio }}</p>
              </div>
            </div>
          </div>

          <!-- Prompt -->
          <div class="space-y-2">
            <h4 class="text-sm font-medium">Prompt</h4>
            <div class="p-3 bg-muted/50 rounded-lg text-sm whitespace-pre-wrap">
              {{ selectedTask.prompt }}
            </div>
          </div>

          <!-- Provider 信息 -->
          <div class="space-y-3">
            <h4 class="text-sm font-medium">Provider 信息</h4>
            <div class="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span class="text-muted-foreground">用户</span>
                <p class="font-medium">{{ selectedTask.username }}</p>
              </div>
              <div>
                <span class="text-muted-foreground">Provider</span>
                <p class="font-medium">{{ selectedTask.provider_name }}</p>
              </div>
              <div>
                <span class="text-muted-foreground">客户端格式</span>
                <p class="font-medium">{{ selectedTask.client_api_format }}</p>
              </div>
              <div>
                <span class="text-muted-foreground">Provider 格式</span>
                <p class="font-medium">{{ selectedTask.provider_api_format }}</p>
              </div>
            </div>
          </div>

          <!-- 候选 Key 追踪 -->
          <div v-if="selectedTask.request_metadata?.candidate_keys?.length" class="space-y-3">
            <h4 class="text-sm font-medium flex items-center gap-2">
              <Key class="w-4 h-4" />
              候选 Key 列表
              <span class="text-xs text-muted-foreground">({{ selectedTask.request_metadata.candidate_keys.length }} 个)</span>
            </h4>
            <div class="space-y-2">
              <div
                v-for="candidateKey in selectedTask.request_metadata.candidate_keys"
                :key="candidateKey.key_id"
                class="p-2 rounded-lg text-xs border"
                :class="candidateKey.selected ? 'bg-primary/10 border-primary/30' : 'bg-muted/30 border-border/60'"
              >
                <div class="flex items-center justify-between">
                  <div class="flex items-center gap-2">
                    <span class="font-medium">{{ candidateKey.provider_name }}</span>
                    <span v-if="candidateKey.key_name" class="text-muted-foreground">/ {{ candidateKey.key_name }}</span>
                    <Badge v-if="candidateKey.selected" variant="default" class="text-[10px] px-1.5 py-0">已选中</Badge>
                    <Badge v-if="candidateKey.has_billing_rule === false" variant="outline" class="text-[10px] px-1.5 py-0 text-amber-500 border-amber-500/50">无计费规则</Badge>
                  </div>
                  <span class="text-muted-foreground">优先级: {{ candidateKey.priority }}</span>
                </div>
                <div class="flex items-center gap-3 mt-1 text-muted-foreground">
                  <span>Auth: {{ candidateKey.auth_type }}</span>
                  <span class="font-mono truncate max-w-[120px]" :title="candidateKey.key_id">Key: {{ candidateKey.key_id.slice(0, 8) }}...</span>
                </div>
              </div>
            </div>
          </div>

          <!-- 请求追踪 -->
          <div v-if="selectedTask.request_metadata" class="space-y-3">
            <h4 class="text-sm font-medium">请求追踪</h4>
            <div class="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span class="text-muted-foreground">Request ID</span>
                <p class="font-mono text-xs truncate" :title="selectedTask.request_metadata.request_id">{{ selectedTask.request_metadata.request_id }}</p>
              </div>
              <div>
                <span class="text-muted-foreground">Client IP</span>
                <p class="font-medium">{{ selectedTask.request_metadata.client_ip }}</p>
              </div>
              <div class="col-span-2">
                <span class="text-muted-foreground">User Agent</span>
                <p class="text-xs truncate" :title="selectedTask.request_metadata.user_agent">{{ selectedTask.request_metadata.user_agent }}</p>
              </div>
            </div>
          </div>

          <!-- 轮询信息 -->
          <div class="space-y-3">
            <h4 class="text-sm font-medium">轮询信息</h4>
            <div class="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span class="text-muted-foreground">轮询次数</span>
                <p class="font-medium">{{ selectedTask.poll_count }} / {{ selectedTask.max_poll_count }}</p>
              </div>
              <div>
                <span class="text-muted-foreground">轮询间隔</span>
                <p class="font-medium">{{ selectedTask.poll_interval_seconds }}s</p>
              </div>
              <div v-if="selectedTask.next_poll_at">
                <span class="text-muted-foreground">下次轮询</span>
                <p class="font-medium">{{ formatDate(selectedTask.next_poll_at) }}</p>
              </div>
            </div>
          </div>

          <!-- 时间信息 -->
          <div class="space-y-3">
            <h4 class="text-sm font-medium">时间信息</h4>
            <div class="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span class="text-muted-foreground">创建时间</span>
                <p class="font-medium">{{ formatDate(selectedTask.created_at) }}</p>
              </div>
              <div v-if="selectedTask.submitted_at">
                <span class="text-muted-foreground">提交时间</span>
                <p class="font-medium">{{ formatDate(selectedTask.submitted_at) }}</p>
              </div>
              <div v-if="selectedTask.completed_at">
                <span class="text-muted-foreground">完成时间</span>
                <p class="font-medium">{{ formatDate(selectedTask.completed_at) }}</p>
              </div>
            </div>
          </div>

          <!-- 视频结果 -->
          <div v-if="selectedTask.status === 'completed' || selectedTask.video_url || selectedTask.video_urls?.length" class="space-y-3">
            <h4 class="text-sm font-medium flex items-center gap-2">
              <Video class="w-4 h-4" />
              视频结果
            </h4>
            
            <!-- 主视频 -->
            <div v-if="selectedTask.video_url" class="space-y-2">
              <video
                :src="selectedTask.video_url"
                controls
                class="w-full rounded-lg"
              />
              <!-- 视频链接 -->
              <div class="p-2 bg-muted/50 rounded text-xs">
                <div class="flex items-center justify-between gap-2">
                  <span class="text-muted-foreground truncate flex-1" :title="selectedTask.video_url">
                    {{ selectedTask.video_url }}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    class="h-6 px-2 text-xs"
                    @click="copyToClipboard(selectedTask.video_url)"
                  >
                    复制链接
                  </Button>
                </div>
              </div>
              <p v-if="selectedTask.video_expires_at" class="text-xs text-muted-foreground">
                过期时间: {{ formatDate(selectedTask.video_expires_at) }}
              </p>
            </div>

            <!-- 多个视频（如果有） -->
            <div v-else-if="selectedTask.video_urls?.length" class="space-y-3">
              <div v-for="(url, index) in selectedTask.video_urls" :key="index" class="space-y-2">
                <p class="text-xs text-muted-foreground">视频 {{ index + 1 }}</p>
                <video :src="url" controls class="w-full rounded-lg" />
                <div class="p-2 bg-muted/50 rounded text-xs">
                  <div class="flex items-center justify-between gap-2">
                    <span class="text-muted-foreground truncate flex-1" :title="url">{{ url }}</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      class="h-6 px-2 text-xs"
                      @click="copyToClipboard(url)"
                    >
                      复制链接
                    </Button>
                  </div>
                </div>
              </div>
            </div>

            <!-- 任务完成但无视频 -->
            <div v-else-if="selectedTask.status === 'completed'" class="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg text-center">
              <p class="text-sm text-amber-600 dark:text-amber-400">任务已完成，但视频链接不可用或已过期</p>
            </div>
          </div>

          <!-- 任务完成响应体 -->
          <div v-if="selectedTask.request_metadata?.poll_raw_response" class="space-y-3">
            <div class="flex items-center justify-between">
              <h4 class="text-sm font-medium flex items-center gap-2">
                <FileJson class="w-4 h-4" />
                任务响应
              </h4>
              <Button
                variant="ghost"
                size="sm"
                class="h-6 px-2 text-xs"
                @click="copyToClipboard(JSON.stringify(selectedTask.request_metadata.poll_raw_response, null, 2))"
              >
                复制
              </Button>
            </div>
            <div class="p-3 bg-muted/50 rounded-lg overflow-x-auto">
              <pre class="text-xs font-mono whitespace-pre-wrap break-all">{{ formatJson(selectedTask.request_metadata.poll_raw_response) }}</pre>
            </div>
          </div>

          <!-- 操作按钮 -->
          <div v-if="canCancel(selectedTask.status)" class="pt-4 border-t">
            <Button
              variant="destructive"
              class="w-full"
              @click="cancelTask(selectedTask)"
            >
              <XCircle class="w-4 h-4 mr-2" />
              取消任务
            </Button>
          </div>
            </div>
          </Card>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>

<style scoped>
.drawer-enter-active,
.drawer-leave-active {
  transition: all 0.3s ease;
}
.drawer-enter-active > div:first-child,
.drawer-leave-active > div:first-child {
  transition: opacity 0.3s ease;
}
.drawer-enter-active > div:last-child,
.drawer-leave-active > div:last-child {
  transition: transform 0.3s ease;
}
.drawer-enter-from,
.drawer-leave-to {
  opacity: 0;
}
.drawer-enter-from > div:last-child,
.drawer-leave-to > div:last-child {
  transform: translateX(100%);
}
</style>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { asyncTasksApi, type AsyncTaskItem, type AsyncTaskDetail, type AsyncTaskStatsResponse, type AsyncTaskStatus } from '@/api/async-tasks'
import { useToast } from '@/composables/useToast'
import Card from '@/components/ui/card.vue'
import Button from '@/components/ui/button.vue'
import Input from '@/components/ui/input.vue'
import Badge from '@/components/ui/badge.vue'
import Select from '@/components/ui/select.vue'
import SelectTrigger from '@/components/ui/select-trigger.vue'
import SelectValue from '@/components/ui/select-value.vue'
import SelectContent from '@/components/ui/select-content.vue'
import SelectItem from '@/components/ui/select-item.vue'
import {
  Zap,
  Video,
  Loader2,
  FileJson,
  CheckCircle,
  Calendar,
  RefreshCw,
  User,
  Server,
  Clock,
  Timer,
  XCircle,
  ChevronRight,
  X,
  Key,
} from 'lucide-vue-next'

const { toast } = useToast()

// 状态
const loading = ref(false)
const tasks = ref<AsyncTaskItem[]>([])
const stats = ref<AsyncTaskStatsResponse | null>(null)
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(20)
const filterStatus = ref('all')
const filterModel = ref('')
const showDetail = ref(false)
const selectedTask = ref<AsyncTaskDetail | null>(null)

const totalPages = computed(() => Math.ceil(total.value / pageSize.value))

// 判断是否为视频任务
function isVideoTask(task: AsyncTaskItem): boolean {
  return task.task_type === 'video' || !!task.video_url || !!task.duration_seconds
}

// 获取任务列表
async function fetchTasks() {
  loading.value = true
  try {
    const response = await asyncTasksApi.list({
      status: filterStatus.value !== 'all' ? filterStatus.value as AsyncTaskStatus : undefined,
      model: filterModel.value || undefined,
      page: currentPage.value,
      page_size: pageSize.value,
    })
    tasks.value = response.items
    total.value = response.total
  } catch (error: any) {
    toast({
      title: '获取任务列表失败',
      description: error.message,
      variant: 'destructive',
    })
  } finally {
    loading.value = false
  }
}

// 获取统计数据
async function fetchStats() {
  try {
    stats.value = await asyncTasksApi.getStats()
  } catch (error) {
    console.error('Failed to fetch stats:', error)
  }
}

// 打开任务详情
async function openTaskDetail(task: AsyncTaskItem) {
  try {
    selectedTask.value = await asyncTasksApi.getDetail(task.id)
    showDetail.value = true
  } catch (error: any) {
    toast({
      title: '获取任务详情失败',
      description: error.message,
      variant: 'destructive',
    })
  }
}

// 取消任务
async function cancelTask(task: AsyncTaskItem | AsyncTaskDetail) {
  if (!confirm('确定要取消这个任务吗？')) return
  try {
    await asyncTasksApi.cancel(task.id)
    toast({
      title: '任务已取消',
    })
    fetchTasks()
    fetchStats()
    if (showDetail.value) {
      showDetail.value = false
      selectedTask.value = null
    }
  } catch (error: any) {
    toast({
      title: '取消任务失败',
      description: error.message,
      variant: 'destructive',
    })
  }
}

// 状态相关
function getStatusVariant(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (status) {
    case 'completed':
      return 'default'
    case 'failed':
      return 'destructive'
    case 'cancelled':
      return 'outline'
    default:
      return 'secondary'
  }
}

function getStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    pending: '待处理',
    submitted: '已提交',
    queued: '排队中',
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
    cancelled: '已取消',
  }
  return labels[status] || status
}

function canCancel(status: string): boolean {
  return ['pending', 'submitted', 'queued', 'processing'].includes(status)
}

// 格式化日期
function formatDate(dateStr: string | null): string {
  if (!dateStr) return '-'
  const date = new Date(dateStr)
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

// 复制到剪贴板
async function copyToClipboard(text: string) {
  try {
    await navigator.clipboard.writeText(text)
    toast({
      title: '已复制到剪贴板',
    })
  } catch (error) {
    toast({
      title: '复制失败',
      variant: 'destructive',
    })
  }
}

// 格式化 JSON
function formatJson(obj: any): string {
  try {
    return JSON.stringify(obj, null, 2)
  } catch {
    return String(obj)
  }
}

// 分页
function goToPage(page: number) {
  currentPage.value = page
  fetchTasks()
}

// 监听筛选条件变化
let filterTimeout: number
watch(filterStatus, () => {
  currentPage.value = 1
  fetchTasks()
})
watch(filterModel, () => {
  clearTimeout(filterTimeout)
  filterTimeout = window.setTimeout(() => {
    currentPage.value = 1
    fetchTasks()
  }, 400)
})

onMounted(() => {
  fetchTasks()
  fetchStats()
})
</script>
