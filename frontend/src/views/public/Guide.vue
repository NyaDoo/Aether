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

        <!-- Spacer -->
        <div class="flex-1" />

        <!-- Right: Actions -->
        <div class="flex items-center gap-1 sm:gap-2 shrink-0">
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

    <!-- Main Content -->
    <main class="relative z-10 px-4 sm:px-8 md:px-16 lg:px-20 py-12 md:py-20">
      <div class="max-w-4xl mx-auto">
        <!-- Page Title -->
        <div class="text-center mb-12">
          <div
            class="inline-flex items-center gap-1.5 md:gap-2 rounded-full bg-[#cc785c]/10 dark:bg-[#cc785c]/20 border border-[#cc785c]/20 dark:border-[#cc785c]/40 px-3 md:px-4 py-1.5 md:py-2 text-xs md:text-sm font-medium text-[#cc785c] dark:text-[#d4a27f] mb-4 md:mb-6 backdrop-blur-sm"
          >
            <BookOpen class="h-3.5 w-3.5 md:h-4 md:w-4" />
            配置教程
          </div>
          <h1 class="text-3xl sm:text-4xl md:text-5xl font-bold text-[#191919] dark:text-white mb-4">
            配置指南
          </h1>
          <p class="text-base md:text-lg text-[#666663] dark:text-[#c9c3b4] max-w-2xl mx-auto">
            快速上手 Aether AI Gateway 的完整配置教程
          </p>
        </div>

        <!-- Wiki Content Placeholder -->
        <div class="bg-white/70 dark:bg-[#262624]/80 backdrop-blur-sm rounded-2xl p-8 md:p-12 border border-[#e5e4df] dark:border-[rgba(227,224,211,0.16)]">
          <div class="text-center py-16">
            <div class="flex h-16 w-16 items-center justify-center rounded-2xl bg-[#cc785c]/10 dark:bg-[#cc785c]/15 mx-auto mb-6">
              <FileText class="h-8 w-8 text-[#cc785c] dark:text-[#d4a27f]" />
            </div>
            <h2 class="text-xl md:text-2xl font-bold text-[#191919] dark:text-white mb-3">
              教程文档即将上线
            </h2>
            <p class="text-[#666663] dark:text-[#c9c3b4] max-w-md mx-auto mb-8">
              我们正在努力编写详细的配置教程，包括 Claude Code、Codex CLI 和 Gemini CLI 的完整配置指南。
            </p>
            <div class="flex flex-wrap items-center justify-center gap-3">
              <div class="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400 text-sm font-medium">
                <Loader2 class="h-4 w-4 animate-spin" />
                文档编写中
              </div>
            </div>
          </div>

          <!-- Coming Soon Sections -->
          <div class="mt-12 pt-8 border-t border-[#e5e4df] dark:border-[rgba(227,224,211,0.12)]">
            <h3 class="text-lg font-semibold text-[#191919] dark:text-white mb-6 text-center">
              即将推出的内容
            </h3>
            <div class="grid md:grid-cols-3 gap-4">
              <div class="p-4 rounded-xl bg-[#fafaf7] dark:bg-[#1f1d1a] border border-[#e5e4df]/50 dark:border-[rgba(227,224,211,0.08)]">
                <div class="flex items-center gap-3 mb-2">
                  <div class="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500/10">
                    <Code2 class="h-4 w-4 text-amber-500" />
                  </div>
                  <span class="font-medium text-[#191919] dark:text-white">Claude Code</span>
                </div>
                <p class="text-sm text-[#666663] dark:text-[#c9c3b4]">
                  IDE 集成配置教程
                </p>
              </div>
              <div class="p-4 rounded-xl bg-[#fafaf7] dark:bg-[#1f1d1a] border border-[#e5e4df]/50 dark:border-[rgba(227,224,211,0.08)]">
                <div class="flex items-center gap-3 mb-2">
                  <div class="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/10">
                    <Terminal class="h-4 w-4 text-emerald-500" />
                  </div>
                  <span class="font-medium text-[#191919] dark:text-white">Codex CLI</span>
                </div>
                <p class="text-sm text-[#666663] dark:text-[#c9c3b4]">
                  命令行工具配置教程
                </p>
              </div>
              <div class="p-4 rounded-xl bg-[#fafaf7] dark:bg-[#1f1d1a] border border-[#e5e4df]/50 dark:border-[rgba(227,224,211,0.08)]">
                <div class="flex items-center gap-3 mb-2">
                  <div class="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                    <Sparkles class="h-4 w-4 text-primary" />
                  </div>
                  <span class="font-medium text-[#191919] dark:text-white">Gemini CLI</span>
                </div>
                <p class="text-sm text-[#666663] dark:text-[#c9c3b4]">
                  多模态 AI 配置教程
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { RouterLink } from 'vue-router'
import {
  BookOpen,
  Code2,
  FileText,
  Loader2,
  Moon,
  Sparkles,
  Sun,
  SunMoon,
  Terminal
} from 'lucide-vue-next'
import GithubIcon from '@/components/icons/GithubIcon.vue'
import HeaderLogo from '@/components/HeaderLogo.vue'
import { useDarkMode } from '@/composables/useDarkMode'

const { themeMode, toggleDarkMode } = useDarkMode()
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

button, nav, a, .inline-flex {
  font-family: var(--sans-serif);
}
</style>
