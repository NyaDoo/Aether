<script setup lang="ts">
import { RouterLink } from 'vue-router'
import { ArrowRight, Layers, Check, Info, Shuffle, TrendingUp, Gauge, Clock } from 'lucide-vue-next'
import { loadBalanceModes, panelClasses } from './guide-config'

const props = withDefaults(
  defineProps<{
    baseUrl?: string
  }>(),
  {
    baseUrl: typeof window !== 'undefined' ? window.location.origin : 'https://your-aether.com'
  }
)

// 模型配置字段
const modelFields = [
  { name: '模型名称', description: '用户调用时使用的模型名，如 gpt-4、claude-3', required: true },
  { name: '别名', description: '模型的其他名称，用户可以用别名调用', required: false },
  { name: '关联端点', description: '该模型使用的端点列表，可多选', required: true },
  { name: '负载均衡模式', description: '多端点时的请求分发策略', required: false },
  { name: '目标模型名', description: '发送给端点的实际模型名（如端点模型名不同）', required: false },
  { name: '是否启用', description: '禁用后用户无法使用该模型', required: false }
]

// 负载均衡图标
const lbIcons = {
  priority: TrendingUp,
  random: Shuffle,
  round_robin: Clock,
  weighted: Gauge,
  latency: Clock
}
</script>

<template>
  <div class="space-y-8">
    <!-- 标题 -->
    <div class="space-y-4">
      <h1 class="text-3xl font-bold text-[#262624] dark:text-[#f1ead8]">
        模型管理
      </h1>
      <p class="text-lg text-[#666663] dark:text-[#a3a094]">
        模型是用户实际调用的对象。通过模型配置，你可以定义模型名称、关联端点、配置负载均衡策略。
      </p>
    </div>

    <!-- 模型概念 -->
    <section class="space-y-4">
      <h2 class="text-xl font-semibold text-[#262624] dark:text-[#f1ead8]">
        什么是模型？
      </h2>

      <div :class="[panelClasses.section, 'p-5']">
        <div class="flex items-center gap-3 mb-4">
          <div class="p-2 rounded-lg bg-purple-500/10">
            <Layers class="h-5 w-5 text-purple-500" />
          </div>
          <h3 class="font-semibold text-[#262624] dark:text-[#f1ead8]">模型 (Model)</h3>
        </div>

        <ul class="space-y-3 text-sm text-[#666663] dark:text-[#a3a094]">
          <li class="flex items-start gap-2">
            <Check class="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span>模型是用户在请求中指定的 <code class="text-xs bg-[#f5f5f0] dark:bg-[#1f1d1a] px-1.5 py-0.5 rounded">model</code> 参数值</span>
          </li>
          <li class="flex items-start gap-2">
            <Check class="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span>一个模型可以关联多个端点，实现负载均衡和故障转移</span>
          </li>
          <li class="flex items-start gap-2">
            <Check class="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span>可以设置别名，让用户用多个名称访问同一个模型</span>
          </li>
          <li class="flex items-start gap-2">
            <Check class="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span>支持模型名映射，请求中的模型名可以与发给端点的不同</span>
          </li>
        </ul>
      </div>

      <div :class="[panelClasses.section, 'p-4']">
        <div class="flex items-start gap-3">
          <Info class="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
          <div class="text-sm text-[#666663] dark:text-[#a3a094]">
            <p class="font-medium text-[#262624] dark:text-[#f1ead8]">示例场景</p>
            <p class="mt-1">
              用户请求模型 <code class="text-xs bg-[#f5f5f0] dark:bg-[#1f1d1a] px-1.5 py-0.5 rounded">gpt-4</code>，
              你可以配置它同时使用 OpenAI 官方端点和某个代理端点。
              系统会根据负载均衡策略选择一个端点，如果失败则自动尝试其他端点。
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- 创建模型 -->
    <section class="space-y-4">
      <h2 class="text-xl font-semibold text-[#262624] dark:text-[#f1ead8]">
        创建模型
      </h2>

      <div :class="[panelClasses.section, 'p-5 space-y-4']">
        <div class="flex items-start gap-4">
          <div class="w-8 h-8 rounded-full bg-[#cc785c] flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
            1
          </div>
          <div>
            <h3 class="font-semibold text-[#262624] dark:text-[#f1ead8]">进入模型管理页面</h3>
            <p class="text-sm text-[#666663] dark:text-[#a3a094] mt-1">
              在管理后台左侧菜单点击「模型管理」
            </p>
          </div>
        </div>

        <div class="flex items-start gap-4">
          <div class="w-8 h-8 rounded-full bg-[#cc785c] flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
            2
          </div>
          <div>
            <h3 class="font-semibold text-[#262624] dark:text-[#f1ead8]">添加模型</h3>
            <p class="text-sm text-[#666663] dark:text-[#a3a094] mt-1">
              点击「添加模型」，填写模型名称（这是用户调用时使用的名称）
            </p>
          </div>
        </div>

        <div class="flex items-start gap-4">
          <div class="w-8 h-8 rounded-full bg-[#cc785c] flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
            3
          </div>
          <div>
            <h3 class="font-semibold text-[#262624] dark:text-[#f1ead8]">关联端点</h3>
            <p class="text-sm text-[#666663] dark:text-[#a3a094] mt-1">
              选择该模型要使用的端点，可以选择多个。如果端点的模型名与你定义的不同，需要设置目标模型名映射。
            </p>
          </div>
        </div>

        <div class="flex items-start gap-4">
          <div class="w-8 h-8 rounded-full bg-[#cc785c] flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
            4
          </div>
          <div>
            <h3 class="font-semibold text-[#262624] dark:text-[#f1ead8]">配置负载均衡</h3>
            <p class="text-sm text-[#666663] dark:text-[#a3a094] mt-1">
              如果选择了多个端点，可以配置负载均衡策略
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- 模型配置字段 -->
    <section class="space-y-4">
      <h2 class="text-xl font-semibold text-[#262624] dark:text-[#f1ead8]">
        配置字段说明
      </h2>

      <div :class="[panelClasses.section, 'overflow-hidden']">
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-[#e5e4df] dark:border-[rgba(227,224,211,0.12)] bg-[#fafaf7]/50 dark:bg-[#1f1d1a]/50">
                <th class="px-4 py-3 text-left font-medium text-[#666663] dark:text-[#a3a094]">字段</th>
                <th class="px-4 py-3 text-left font-medium text-[#666663] dark:text-[#a3a094]">说明</th>
                <th class="px-4 py-3 text-center font-medium text-[#666663] dark:text-[#a3a094]">必填</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="field in modelFields"
                :key="field.name"
                class="border-b border-[#e5e4df] dark:border-[rgba(227,224,211,0.08)] last:border-0"
              >
                <td class="px-4 py-3 font-medium text-[#262624] dark:text-[#f1ead8]">
                  {{ field.name }}
                </td>
                <td class="px-4 py-3 text-[#666663] dark:text-[#a3a094]">
                  {{ field.description }}
                </td>
                <td class="px-4 py-3 text-center">
                  <span v-if="field.required" :class="panelClasses.badgeGreen">必填</span>
                  <span v-else class="text-[#999]">可选</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- 负载均衡模式 -->
    <section class="space-y-4">
      <h2 class="text-xl font-semibold text-[#262624] dark:text-[#f1ead8]">
        负载均衡模式
      </h2>
      <p class="text-[#666663] dark:text-[#a3a094]">
        当模型关联多个端点时，系统会根据负载均衡模式选择端点：
      </p>

      <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <div
          v-for="lb in loadBalanceModes"
          :key="lb.mode"
          :class="[panelClasses.section, 'p-4']"
        >
          <div class="flex items-center gap-3 mb-2">
            <component
              :is="lbIcons[lb.mode as keyof typeof lbIcons] || Shuffle"
              class="h-5 w-5 text-[#cc785c]"
            />
            <h3 class="font-semibold text-[#262624] dark:text-[#f1ead8]">{{ lb.name }}</h3>
          </div>
          <p class="text-sm text-[#666663] dark:text-[#a3a094]">{{ lb.description }}</p>
        </div>
      </div>

      <div :class="[panelClasses.section, 'p-4']">
        <div class="flex items-start gap-3">
          <Info class="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
          <div class="text-sm text-[#666663] dark:text-[#a3a094]">
            <p class="font-medium text-[#262624] dark:text-[#f1ead8]">推荐配置</p>
            <p class="mt-1">
              <strong>主备切换场景</strong>：使用「优先级」模式，将官方端点设为高优先级，代理端点设为低优先级。<br />
              <strong>分摊负载场景</strong>：使用「轮询」或「加权」模式，将多个同质端点平均分配请求。
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- 模型名映射 -->
    <section class="space-y-4">
      <h2 class="text-xl font-semibold text-[#262624] dark:text-[#f1ead8]">
        模型名映射
      </h2>

      <div :class="[panelClasses.section, 'p-5']">
        <p class="text-sm text-[#666663] dark:text-[#a3a094] mb-4">
          有时候你希望用户调用的模型名与实际发给端点的模型名不同。例如：
        </p>

        <div class="space-y-3">
          <div class="flex items-center gap-4 text-sm">
            <div class="px-3 py-2 rounded-lg bg-purple-500/10 border border-purple-500/20">
              <span class="text-purple-600 dark:text-purple-400">用户请求</span>
              <code class="ml-2 text-xs bg-[#f5f5f0] dark:bg-[#1f1d1a] px-1.5 py-0.5 rounded">gpt-4</code>
            </div>
            <ArrowRight class="h-4 w-4 text-[#999]" />
            <div class="px-3 py-2 rounded-lg bg-green-500/10 border border-green-500/20">
              <span class="text-green-600 dark:text-green-400">发给端点</span>
              <code class="ml-2 text-xs bg-[#f5f5f0] dark:bg-[#1f1d1a] px-1.5 py-0.5 rounded">gpt-4-turbo-preview</code>
            </div>
          </div>

          <div class="flex items-center gap-4 text-sm">
            <div class="px-3 py-2 rounded-lg bg-purple-500/10 border border-purple-500/20">
              <span class="text-purple-600 dark:text-purple-400">用户请求</span>
              <code class="ml-2 text-xs bg-[#f5f5f0] dark:bg-[#1f1d1a] px-1.5 py-0.5 rounded">claude-3</code>
            </div>
            <ArrowRight class="h-4 w-4 text-[#999]" />
            <div class="px-3 py-2 rounded-lg bg-green-500/10 border border-green-500/20">
              <span class="text-green-600 dark:text-green-400">发给端点</span>
              <code class="ml-2 text-xs bg-[#f5f5f0] dark:bg-[#1f1d1a] px-1.5 py-0.5 rounded">claude-3-opus-20240229</code>
            </div>
          </div>
        </div>

        <p class="text-sm text-[#666663] dark:text-[#a3a094] mt-4">
          在模型的端点关联配置中，可以为每个端点设置「目标模型名」来实现这种映射。
        </p>
      </div>
    </section>

    <!-- 别名功能 -->
    <section class="space-y-4">
      <h2 class="text-xl font-semibold text-[#262624] dark:text-[#f1ead8]">
        模型别名
      </h2>

      <div :class="[panelClasses.section, 'p-5']">
        <p class="text-sm text-[#666663] dark:text-[#a3a094] mb-4">
          别名让用户可以用多个名称访问同一个模型。适用于：
        </p>

        <ul class="space-y-2 text-sm text-[#666663] dark:text-[#a3a094]">
          <li class="flex items-start gap-2">
            <Check class="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span>兼容旧的模型名（如 <code class="text-xs bg-[#f5f5f0] dark:bg-[#1f1d1a] px-1.5 py-0.5 rounded">gpt-4-turbo</code> 和 <code class="text-xs bg-[#f5f5f0] dark:bg-[#1f1d1a] px-1.5 py-0.5 rounded">gpt-4-turbo-preview</code> 指向同一个模型）</span>
          </li>
          <li class="flex items-start gap-2">
            <Check class="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span>提供简短易记的名称（如用 <code class="text-xs bg-[#f5f5f0] dark:bg-[#1f1d1a] px-1.5 py-0.5 rounded">claude</code> 代替完整版本号）</span>
          </li>
          <li class="flex items-start gap-2">
            <Check class="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span>版本迁移时保持向后兼容</span>
          </li>
        </ul>
      </div>
    </section>

    <!-- 下一步 -->
    <section class="pt-4">
      <RouterLink
        to="/guide/user-key"
        :class="[panelClasses.section, panelClasses.cardHover, 'p-4 flex items-center gap-3 group']"
      >
        <div class="flex-1">
          <div class="font-medium text-[#262624] dark:text-[#f1ead8]">下一步：用户与密钥</div>
          <div class="text-sm text-[#666663] dark:text-[#a3a094]">管理用户和 API Key</div>
        </div>
        <ArrowRight class="h-5 w-5 text-[#999] group-hover:text-[#cc785c] transition-colors" />
      </RouterLink>
    </section>
  </div>
</template>
