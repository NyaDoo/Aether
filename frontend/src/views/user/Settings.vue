<template>
  <div class="container mx-auto px-4 py-8">
    <h2 class="text-2xl font-bold text-foreground mb-6">
      个人设置
    </h2>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- 左侧：个人信息和密码 -->
      <div class="lg:col-span-2 space-y-6">
        <!-- 基本信息 -->
        <Card class="p-6">
          <h3 class="text-lg font-medium text-foreground mb-4">
            基本信息
          </h3>
          <form
            class="space-y-4"
            @submit.prevent="updateProfile"
          >
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label for="username">用户名</Label>
                <Input
                  id="username"
                  v-model="profileForm.username"
                  class="mt-1"
                />
              </div>
              <div>
                <Label for="email">邮箱</Label>
                <Input
                  id="email"
                  v-model="profileForm.email"
                  type="email"
                  class="mt-1"
                />
              </div>
            </div>

            <div>
              <Label for="bio">个人简介</Label>
              <Textarea
                id="bio"
                v-model="preferencesForm.bio"
                rows="3"
                class="mt-1"
              />
            </div>

            <div>
              <Label for="avatar">头像 URL</Label>
              <Input
                id="avatar"
                v-model="preferencesForm.avatar_url"
                type="url"
                class="mt-1"
              />
              <p class="mt-1 text-sm text-muted-foreground">
                输入头像图片的 URL 地址
              </p>
            </div>

            <Button
              type="submit"
              :disabled="savingProfile"
              class="shadow-none hover:shadow-none"
            >
              {{ savingProfile ? '保存中...' : '保存修改' }}
            </Button>
          </form>
        </Card>

        <!-- 修改密码 -->
        <Card class="p-6">
          <h3 class="text-lg font-medium text-foreground mb-4">
            修改密码
          </h3>
          <form
            class="space-y-4"
            @submit.prevent="changePassword"
          >
            <div>
              <Label for="old-password">当前密码</Label>
              <Input
                id="old-password"
                v-model="passwordForm.old_password"
                type="password"
                class="mt-1"
              />
            </div>
            <div>
              <Label for="new-password">新密码</Label>
              <Input
                id="new-password"
                v-model="passwordForm.new_password"
                type="password"
                class="mt-1"
              />
            </div>
            <div>
              <Label for="confirm-password">确认新密码</Label>
              <Input
                id="confirm-password"
                v-model="passwordForm.confirm_password"
                type="password"
                class="mt-1"
              />
            </div>
            <Button
              type="submit"
              :disabled="changingPassword"
              class="shadow-none hover:shadow-none"
            >
              {{ changingPassword ? '修改中...' : '修改密码' }}
            </Button>
          </form>
        </Card>

        <!-- 偏好设置 -->
        <Card class="p-6">
          <h3 class="text-lg font-medium text-foreground mb-4">
            偏好设置
          </h3>
          <div class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label for="theme">主题</Label>
                <Select
                  v-model="preferencesForm.theme"
                  v-model:open="themeSelectOpen"
                  @update:model-value="handleThemeChange"
                >
                  <SelectTrigger
                    id="theme"
                    class="mt-1"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="light">
                      浅色
                    </SelectItem>
                    <SelectItem value="dark">
                      深色
                    </SelectItem>
                    <SelectItem value="system">
                      跟随系统
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label for="language">语言</Label>
                <Select
                  v-model="preferencesForm.language"
                  v-model:open="languageSelectOpen"
                  @update:model-value="handleLanguageChange"
                >
                  <SelectTrigger
                    id="language"
                    class="mt-1"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="zh-CN">
                      简体中文
                    </SelectItem>
                    <SelectItem value="en">
                      English
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label for="timezone">时区</Label>
                <Input
                  id="timezone"
                  v-model="preferencesForm.timezone"
                  placeholder="Asia/Shanghai"
                  class="mt-1"
                />
              </div>
            </div>

            <div class="space-y-3">
              <h4 class="font-medium text-foreground">
                通知设置
              </h4>
              <div class="space-y-3">
                <div class="flex items-center justify-between py-2 border-b border-border/40 last:border-0">
                  <div class="flex-1">
                    <Label
                      for="email-notifications"
                      class="text-sm font-medium cursor-pointer"
                    >
                      邮件通知
                    </Label>
                    <p class="text-xs text-muted-foreground mt-1">
                      接收系统重要通知
                    </p>
                  </div>
                  <Switch
                    id="email-notifications"
                    v-model="preferencesForm.notifications.email"
                    @update:model-value="updatePreferences"
                  />
                </div>
                <div class="flex items-center justify-between py-2 border-b border-border/40 last:border-0">
                  <div class="flex-1">
                    <Label
                      for="usage-alerts"
                      class="text-sm font-medium cursor-pointer"
                    >
                      使用提醒
                    </Label>
                    <p class="text-xs text-muted-foreground mt-1">
                      当接近配额限制时提醒
                    </p>
                  </div>
                  <Switch
                    id="usage-alerts"
                    v-model="preferencesForm.notifications.usage_alerts"
                    @update:model-value="updatePreferences"
                  />
                </div>
                <div class="flex items-center justify-between py-2">
                  <div class="flex-1">
                    <Label
                      for="announcement-notifications"
                      class="text-sm font-medium cursor-pointer"
                    >
                      公告通知
                    </Label>
                    <p class="text-xs text-muted-foreground mt-1">
                      接收系统公告
                    </p>
                  </div>
                  <Switch
                    id="announcement-notifications"
                    v-model="preferencesForm.notifications.announcements"
                    @update:model-value="updatePreferences"
                  />
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>

      <!-- 右侧：账户信息和使用量 -->
      <div class="space-y-6">
        <!-- 账户信息 -->
        <Card class="p-6">
          <h3 class="text-lg font-medium text-foreground mb-4">
            账户信息
          </h3>
          <div class="space-y-3">
            <div class="flex justify-between">
              <span class="text-muted-foreground">角色</span>
              <Badge :variant="profile?.role === 'admin' ? 'default' : 'secondary'">
                {{ profile?.role === 'admin' ? '管理员' : '普通用户' }}
              </Badge>
            </div>
            <div class="flex justify-between">
              <span class="text-muted-foreground">账户状态</span>
              <span :class="profile?.is_active ? 'text-success' : 'text-destructive'">
                {{ profile?.is_active ? '活跃' : '已停用' }}
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-muted-foreground">注册时间</span>
              <span class="text-foreground">
                {{ formatDate(profile?.created_at) }}
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-muted-foreground">最后登录</span>
              <span class="text-foreground">
                {{ profile?.last_login_at ? formatDate(profile.last_login_at) : '未记录' }}
              </span>
            </div>
          </div>
        </Card>

        <!-- 使用配额 -->
        <Card class="p-6">
          <h3 class="text-lg font-medium text-foreground mb-4">
            使用配额
          </h3>
          <div class="space-y-4">
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span class="text-muted-foreground">配额使用(美元)</span>
                <span class="text-foreground">
                  <template v-if="isUnlimitedQuota()">
                    {{ formatCurrency(profile?.used_usd || 0) }} /
                    <span class="text-warning">无限制</span>
                  </template>
                  <template v-else>
                    {{ formatCurrency(profile?.used_usd || 0) }} /
                    {{ formatCurrency(profile?.quota_usd || 0) }}
                  </template>
                </span>
              </div>
              <div class="w-full bg-muted rounded-full h-2.5">
                <div
                  class="bg-success h-2.5 rounded-full"
                  :style="`width: ${getUsagePercentage()}%`"
                />
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { meApi, type Profile } from '@/api/me'
import { useDarkMode, type ThemeMode } from '@/composables/useDarkMode'
import Card from '@/components/ui/card.vue'
import Button from '@/components/ui/button.vue'
import Badge from '@/components/ui/badge.vue'
import Input from '@/components/ui/input.vue'
import Label from '@/components/ui/label.vue'
import Textarea from '@/components/ui/textarea.vue'
import Select from '@/components/ui/select.vue'
import SelectTrigger from '@/components/ui/select-trigger.vue'
import SelectValue from '@/components/ui/select-value.vue'
import SelectContent from '@/components/ui/select-content.vue'
import SelectItem from '@/components/ui/select-item.vue'
import Switch from '@/components/ui/switch.vue'
import { useToast } from '@/composables/useToast'
import { formatCurrency } from '@/utils/format'
import { log } from '@/utils/logger'

const authStore = useAuthStore()
const { success, error: showError } = useToast()
const { setThemeMode } = useDarkMode()

const profile = ref<Profile | null>(null)

const profileForm = ref({
  email: '',
  username: ''
})

const passwordForm = ref({
  old_password: '',
  new_password: '',
  confirm_password: ''
})

const preferencesForm = ref({
  avatar_url: '',
  bio: '',
  theme: 'light',
  language: 'zh-CN',
  timezone: 'Asia/Shanghai',
  notifications: {
    email: true,
    usage_alerts: true,
    announcements: true
  }
})

const savingProfile = ref(false)
const changingPassword = ref(false)
const themeSelectOpen = ref(false)
const languageSelectOpen = ref(false)

function handleThemeChange(value: string) {
  preferencesForm.value.theme = value
  themeSelectOpen.value = false
  updatePreferences()

  // 使用 useDarkMode 统一切换主题
  setThemeMode(value as ThemeMode)
}

function handleLanguageChange(value: string) {
  preferencesForm.value.language = value
  languageSelectOpen.value = false
  updatePreferences()
}

onMounted(async () => {
  await loadProfile()
  await loadPreferences()
})

async function loadProfile() {
  try {
    profile.value = await meApi.getProfile()
    profileForm.value = {
      email: profile.value.email,
      username: profile.value.username
    }
  } catch (error) {
    log.error('加载个人信息失败:', error)
    showError('加载个人信息失败')
  }
}

async function loadPreferences() {
  try {
    const prefs = await meApi.getPreferences()

    // 主题以本地 localStorage 为准（useDarkMode 在应用启动时已初始化）
    // 这样可以避免刷新页面时主题被服务端旧值覆盖
    const { themeMode: currentThemeMode } = useDarkMode()
    const localTheme = currentThemeMode.value

    preferencesForm.value = {
      avatar_url: prefs.avatar_url || '',
      bio: prefs.bio || '',
      theme: localTheme,  // 使用本地主题，而非服务端返回值
      language: prefs.language || 'zh-CN',
      timezone: prefs.timezone || 'Asia/Shanghai',
      notifications: {
        email: prefs.notifications?.email ?? true,
        usage_alerts: prefs.notifications?.usage_alerts ?? true,
        announcements: prefs.notifications?.announcements ?? true
      }
    }

    // 如果本地主题和服务端不一致，同步到服务端（静默更新，不提示用户）
    const serverTheme = prefs.theme || 'light'
    if (localTheme !== serverTheme) {
      meApi.updatePreferences({ theme: localTheme }).catch(() => {
        // 静默失败，不影响用户体验
      })
    }
  } catch (error) {
    log.error('加载偏好设置失败:', error)
  }
}

async function updateProfile() {
  savingProfile.value = true
  try {
    await meApi.updateProfile(profileForm.value)

    // 同时更新偏好设置中的 avatar_url 和 bio
    await meApi.updatePreferences({
      avatar_url: preferencesForm.value.avatar_url || undefined,
      bio: preferencesForm.value.bio || undefined,
      theme: preferencesForm.value.theme,
      language: preferencesForm.value.language,
      timezone: preferencesForm.value.timezone || undefined,
      notifications: {
        email: preferencesForm.value.notifications.email,
        usage_alerts: preferencesForm.value.notifications.usage_alerts,
        announcements: preferencesForm.value.notifications.announcements
      }
    })

    success('个人信息已更新')
    await loadProfile()
    authStore.fetchCurrentUser()
  } catch (error) {
    log.error('更新个人信息失败:', error)
    showError('更新个人信息失败')
  } finally {
    savingProfile.value = false
  }
}

async function changePassword() {
  if (passwordForm.value.new_password !== passwordForm.value.confirm_password) {
    showError('两次输入的密码不一致')
    return
  }

  if (passwordForm.value.new_password.length < 8) {
    showError('密码长度至少8位')
    return
  }

  changingPassword.value = true
  try {
    await meApi.changePassword({
      old_password: passwordForm.value.old_password,
      new_password: passwordForm.value.new_password
    })
    success('密码修改成功')
    passwordForm.value = {
      old_password: '',
      new_password: '',
      confirm_password: ''
    }
  } catch (error) {
    log.error('修改密码失败:', error)
    showError('修改密码失败，请检查当前密码是否正确')
  } finally {
    changingPassword.value = false
  }
}

async function updatePreferences() {
  try {
    await meApi.updatePreferences({
      avatar_url: preferencesForm.value.avatar_url || undefined,
      bio: preferencesForm.value.bio || undefined,
      theme: preferencesForm.value.theme,
      language: preferencesForm.value.language,
      timezone: preferencesForm.value.timezone || undefined,
      notifications: {
        email: preferencesForm.value.notifications.email,
        usage_alerts: preferencesForm.value.notifications.usage_alerts,
        announcements: preferencesForm.value.notifications.announcements
      }
    })
    success('设置已保存')
  } catch (error) {
    log.error('更新偏好设置失败:', error)
    showError('保存设置失败')
  }
}

function getUsagePercentage(): number {
  if (!profile.value) return 0

  const quota = profile.value.quota_usd
  const used = profile.value.used_usd
  if (quota == null || quota === 0) return 0
  return Math.min(100, (used / quota) * 100)
}

function isUnlimitedQuota(): boolean {
  return profile.value?.quota_usd == null
}

function formatDate(dateString?: string): string {
  if (!dateString) return '未知'
  return new Date(dateString).toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}
</script>
