<template>
  <div class="min-h-screen literary-grid literary-paper">
    <!-- Header -->
    <header class="sticky top-0 z-50 border-b border-[#cc785c]/10 dark:border-[rgba(227,224,211,0.12)] bg-[#fafaf7]/90 dark:bg-[#191714]/95 backdrop-blur-xl transition-all">
      <div class="h-14 sm:h-16 flex items-center px-3 sm:px-4 md:px-8">
        <!-- Left: Logo & Brand -->
        <RouterLink
          to="/"
          class="flex items-center gap-2 sm:gap-3 group/logo cursor-pointer shrink-0"
        >
          <HeaderLogo
            size="h-7 w-7 sm:h-9 sm:w-9"
            class-name="text-[#191919] dark:text-white"
          />
          <div class="flex flex-col justify-center">
            <h1 class="text-base sm:text-lg font-bold text-[#191919] dark:text-white leading-none">
              Aether
            </h1>
            <span class="text-[9px] sm:text-[10px] text-[#91918d] dark:text-muted-foreground leading-none mt-1 sm:mt-1.5 font-medium tracking-wide">AI Gateway</span>
          </div>
        </RouterLink>

        <!-- Center: Breadcrumb -->
        <div class="hidden md:flex items-center gap-2 ml-8">
          <RouterLink
            to="/guide"
            class="text-sm text-[#666663] dark:text-muted-foreground hover:text-[#191919] dark:hover:text-white transition"
          >
            教程文档
          </RouterLink>
          <ChevronRight
            v-if="currentNavItem && currentNavItem.id !== 'overview'"
            class="h-4 w-4 text-[#91918d]"
          />
          <span
            v-if="currentNavItem && currentNavItem.id !== 'overview'"
            class="text-sm font-medium text-[#191919] dark:text-white"
          >
            {{ currentNavItem.name }}
          </span>
        </div>

        <!-- Spacer -->
        <div class="flex-1" />

        <!-- Right: Actions -->
        <div class="flex items-center gap-1 sm:gap-2 shrink-0">
          <!-- Mobile menu button -->
          <button
            class="md:hidden flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50 transition"
            @click="showMobileNav = !showMobileNav"
          >
            <Menu class="h-4 w-4" />
          </button>

          <!-- Theme Toggle + GitHub Icons -->
          <div class="flex items-center gap-0.5 sm:gap-1">
            <button
              class="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50 transition"
              :title="themeMode === 'system' ? '跟随系统' : themeMode === 'dark' ? '深色模式' : '浅色模式'"
              @click="toggleDarkMode"
            >
              <SunMoon
                v-if="themeMode === 'system'"
                class="h-3.5 w-3.5 sm:h-4 sm:w-4"
              />
              <Sun
                v-else-if="themeMode === 'light'"
                class="h-3.5 w-3.5 sm:h-4 sm:w-4"
              />
              <Moon
                v-else
                class="h-3.5 w-3.5 sm:h-4 sm:w-4"
              />
            </button>
            <a
              href="https://github.com/fawney19/Aether"
              target="_blank"
              rel="noopener noreferrer"
              class="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted/50 transition"
              title="GitHub 仓库"
            >
              <GithubIcon class="h-3.5 w-3.5 sm:h-4 sm:w-4" />
            </a>
          </div>
        </div>
      </div>
    </header>

    <!-- Mobile Nav Overlay -->
    <Transition name="fade">
      <div
        v-if="showMobileNav"
        class="fixed inset-0 z-40 bg-black/50 md:hidden"
        @click="showMobileNav = false"
      />
    </Transition>

    <!-- Mobile Nav Drawer -->
    <Transition name="slide-left">
      <div
        v-if="showMobileNav"
        class="fixed left-0 top-14 bottom-0 z-50 w-64 bg-[#fafaf7] dark:bg-[#191714] border-r border-[#e5e4df] dark:border-[rgba(227,224,211,0.12)] md:hidden overflow-y-auto"
      >
        <nav class="p-4 space-y-1">
          <RouterLink
            v-for="item in guideNavItems"
            :key="item.id"
            :to="item.path"
            class="flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors"
            :class="isActive(item.path)
              ? 'bg-[#cc785c]/10 text-[#cc785c] dark:text-[#d4a27f]'
              : 'text-[#666663] dark:text-muted-foreground hover:bg-[#f0f0eb] dark:hover:bg-[#262624]'"
            @click="showMobileNav = false"
          >
            <component
              :is="item.icon"
              class="h-4 w-4 shrink-0"
            />
            <span class="text-sm font-medium">{{ item.name }}</span>
          </RouterLink>
        </nav>
      </div>
    </Transition>

    <!-- Main Content -->
    <div class="flex">
      <!-- Desktop Sidebar -->
      <aside class="hidden md:block w-64 shrink-0 border-r border-[#e5e4df] dark:border-[rgba(227,224,211,0.12)] bg-[#fafaf7]/50 dark:bg-[#191714]/50">
        <div class="sticky top-16 h-[calc(100vh-4rem)] overflow-y-auto">
          <nav class="p-4 space-y-1">
            <RouterLink
              v-for="item in guideNavItems"
              :key="item.id"
              :to="item.path"
              class="flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors group"
              :class="isActive(item.path)
                ? 'bg-[#cc785c]/10 text-[#cc785c] dark:text-[#d4a27f]'
                : 'text-[#666663] dark:text-muted-foreground hover:bg-[#f0f0eb] dark:hover:bg-[#262624]'"
            >
              <component
                :is="item.icon"
                class="h-4 w-4 shrink-0"
              />
              <div class="flex flex-col">
                <span class="text-sm font-medium">{{ item.name }}</span>
                <span
                  v-if="item.description"
                  class="text-xs text-[#91918d] dark:text-muted-foreground/70"
                >
                  {{ item.description }}
                </span>
              </div>
            </RouterLink>
          </nav>

          <!-- Base URL Input -->
          <div class="p-4 border-t border-[#e5e4df] dark:border-[rgba(227,224,211,0.12)]">
            <label class="block text-xs font-medium text-[#666663] dark:text-muted-foreground mb-2">
              Aether Base URL
            </label>
            <input
              v-model="baseUrl"
              type="text"
              class="w-full px-3 py-2 text-sm rounded-lg border border-[#e5e4df] dark:border-[rgba(227,224,211,0.12)] bg-white dark:bg-[#1f1d1a] text-[#191919] dark:text-white placeholder-[#91918d] focus:outline-none focus:ring-2 focus:ring-[#cc785c]/30"
              placeholder="https://your-aether.com"
            />
            <p class="mt-1.5 text-xs text-[#91918d]">
              代码示例将使用此 URL
            </p>
          </div>
        </div>
      </aside>

      <!-- Page Content -->
      <main class="flex-1 min-w-0">
        <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12">
          <RouterView v-slot="{ Component }">
            <component :is="Component" :base-url="baseUrl" />
          </RouterView>
        </div>
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import {
  ChevronRight,
  Menu,
  Moon,
  Sun,
  SunMoon
} from 'lucide-vue-next'
import GithubIcon from '@/components/icons/GithubIcon.vue'
import HeaderLogo from '@/components/HeaderLogo.vue'
import { useDarkMode } from '@/composables/useDarkMode'
import { guideNavItems } from './guide-config'

const route = useRoute()
const { themeMode, toggleDarkMode } = useDarkMode()

const showMobileNav = ref(false)
const baseUrl = ref(typeof window !== 'undefined' ? window.location.origin : 'https://your-aether.com')

const currentNavItem = computed(() => {
  return guideNavItems.find(item => item.path === route.path)
})

function isActive(path: string): boolean {
  if (path === '/guide') {
    return route.path === '/guide'
  }
  return route.path.startsWith(path)
}
</script>

<style scoped>
/* Typography */
h1, h2, h3 {
  font-family: var(--serif);
  letter-spacing: -0.02em;
  font-weight: 500;
}

p {
  font-family: var(--serif);
  letter-spacing: 0.01em;
  line-height: 1.7;
}

button, nav, a, .inline-flex, input, label {
  font-family: var(--sans-serif);
}

/* Transitions */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.slide-left-enter-active,
.slide-left-leave-active {
  transition: transform 0.2s ease;
}

.slide-left-enter-from,
.slide-left-leave-to {
  transform: translateX(-100%);
}
</style>
