<script setup lang="ts">
import { RouterLink } from 'vue-router'
import { ArrowRight, Shuffle, FileCode, Globe, Shield, Check, Info, AlertTriangle, Settings } from 'lucide-vue-next'
import { panelClasses } from './guide-config'

const props = withDefaults(
  defineProps<{
    baseUrl?: string
  }>(),
  {
    baseUrl: typeof window !== 'undefined' ? window.location.origin : 'https://your-aether.com'
  }
)

// 格式转换示例
const conversionExamples = [
  {
    from: 'OpenAI SDK',
    to: 'Claude API',
    description: '用 OpenAI SDK 调用 Claude 模型',
    icon: 'openai'
  },
  {
    from: 'Anthropic SDK',
    to: 'OpenAI API',
    description: '用 Claude SDK 调用 GPT 模型',
    icon: 'claude'
  },
  {
    from: 'Claude Code',
    to: 'OpenAI API',
    description: 'Claude Code 使用 GPT 模型',
    icon: 'cli'
  }
]

// 请求头规则类型
const headerRuleTypes = [
  { type: 'add', name: '添加', description: '添加新的请求头，如果已存在则跳过' },
  { type: 'set', name: '设置', description: '设置请求头的值，会覆盖已有值' },
  { type: 'remove', name: '删除', description: '删除指定的请求头' }
]

// 系统设置分类
const systemSettings = [
  {
    category: '基础设置',
    items: [
      { name: '系统名称', description: '显示在页面标题和 Logo 旁边的名称' },
      { name: '注册开关', description: '是否允许新用户注册' },
      { name: '默认用户角色', description: '新注册用户的默认角色' }
    ]
  },
  {
    category: 'API 设置',
    items: [
      { name: '格式转换', description: '是否启用跨格式 API 调用' },
      { name: '请求超时', description: '默认的 API 请求超时时间' },
      { name: '流式响应', description: '是否默认启用流式响应' }
    ]
  },
  {
    category: '安全设置',
    items: [
      { name: 'IP 限流', description: '单 IP 的请求频率限制' },
      { name: 'Key 限流', description: '单 Key 的请求频率限制' },
      { name: 'IP 黑名单', description: '全局 IP 黑名单' }
    ]
  }
]
</script>

<template>
  <div class="space-y-8">
    <!-- 标题 -->
    <div class="space-y-4">
      <h1 class="text-3xl font-bold text-[#262624] dark:text-[#f1ead8]">
        高级功能
      </h1>
      <p class="text-lg text-[#666663] dark:text-[#a3a094]">
        Aether 提供了一些高级功能，帮助你实现更灵活的 API 管理。
      </p>
    </div>

    <!-- 格式转换 -->
    <section class="space-y-4">
      <h2 class="text-xl font-semibold text-[#262624] dark:text-[#f1ead8]">
        格式转换
      </h2>

      <div :class="[panelClasses.section, 'p-5']">
        <div class="flex items-center gap-3 mb-4">
          <div class="p-2 rounded-lg bg-purple-500/10">
            <Shuffle class="h-5 w-5 text-purple-500" />
          </div>
          <h3 class="font-semibold text-[#262624] dark:text-[#f1ead8]">什么是格式转换？</h3>
        </div>

        <p class="text-sm text-[#666663] dark:text-[#a3a094] mb-4">
          格式转换允许你用一种 SDK 调用另一种格式的 API。例如用 OpenAI SDK 调用 Claude 模型，
          系统会自动转换请求格式和响应格式。
        </p>

        <div class="grid gap-3 md:grid-cols-3">
          <div
            v-for="example in conversionExamples"
            :key="example.from"
            class="p-4 rounded-lg bg-[#f5f5f0]/50 dark:bg-[#1f1d1a]/50"
          >
            <div class="flex items-center gap-2 mb-2">
              <span :class="panelClasses.badgeBlue">{{ example.from }}</span>
              <ArrowRight class="h-4 w-4 text-[#999]" />
              <span :class="panelClasses.badgeGreen">{{ example.to }}</span>
            </div>
            <p class="text-xs text-[#666663] dark:text-[#a3a094]">{{ example.description }}</p>
          </div>
        </div>
      </div>

      <!-- 如何启用 -->
      <div :class="[panelClasses.section, 'p-5 space-y-4']">
        <h3 class="font-semibold text-[#262624] dark:text-[#f1ead8]">如何启用格式转换</h3>

        <div class="flex items-start gap-4">
          <div class="w-8 h-8 rounded-full bg-[#cc785c] flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
            1
          </div>
          <div>
            <p class="font-medium text-[#262624] dark:text-[#f1ead8]">开启系统设置</p>
            <p class="text-sm text-[#666663] dark:text-[#a3a094] mt-1">
              在「系统设置」中开启 <code class="text-xs bg-[#f5f5f0] dark:bg-[#1f1d1a] px-1.5 py-0.5 rounded">ENABLE_API_FORMAT_CONVERSION</code>
            </p>
          </div>
        </div>

        <div class="flex items-start gap-4">
          <div class="w-8 h-8 rounded-full bg-[#cc785c] flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
            2
          </div>
          <div>
            <p class="font-medium text-[#262624] dark:text-[#f1ead8]">配置端点</p>
            <p class="text-sm text-[#666663] dark:text-[#a3a094] mt-1">
              在端点配置中，启用「格式接受配置」并选择接受的入站格式
            </p>
          </div>
        </div>

        <div class="flex items-start gap-4">
          <div class="w-8 h-8 rounded-full bg-[#cc785c] flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
            3
          </div>
          <div>
            <p class="font-medium text-[#262624] dark:text-[#f1ead8]">使用</p>
            <p class="text-sm text-[#666663] dark:text-[#a3a094] mt-1">
              用户现在可以用 OpenAI SDK 调用配置在 Claude 格式端点上的模型
            </p>
          </div>
        </div>
      </div>

      <div :class="[panelClasses.section, 'p-4']">
        <div class="flex items-start gap-3">
          <Info class="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
          <div class="text-sm text-[#666663] dark:text-[#a3a094]">
            <p class="font-medium text-[#262624] dark:text-[#f1ead8]">注意事项</p>
            <p class="mt-1">
              格式转换会增加少量延迟（通常 &lt;10ms）。部分特有功能（如 Claude 的 thinking、OpenAI 的 function calling）
              可能无法完美转换，建议在实际场景中测试。
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- 请求头规则 -->
    <section class="space-y-4">
      <h2 class="text-xl font-semibold text-[#262624] dark:text-[#f1ead8]">
        请求头规则
      </h2>

      <div :class="[panelClasses.section, 'p-5']">
        <div class="flex items-center gap-3 mb-4">
          <div class="p-2 rounded-lg bg-green-500/10">
            <FileCode class="h-5 w-5 text-green-500" />
          </div>
          <h3 class="font-semibold text-[#262624] dark:text-[#f1ead8]">什么是请求头规则？</h3>
        </div>

        <p class="text-sm text-[#666663] dark:text-[#a3a094] mb-4">
          请求头规则允许你在转发请求时修改 HTTP 请求头。可以添加、修改或删除特定的请求头。
        </p>

        <div :class="[panelClasses.section, 'overflow-hidden']">
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="border-b border-[#e5e4df] dark:border-[rgba(227,224,211,0.12)] bg-[#fafaf7]/50 dark:bg-[#1f1d1a]/50">
                  <th class="px-4 py-3 text-left font-medium text-[#666663] dark:text-[#a3a094]">类型</th>
                  <th class="px-4 py-3 text-left font-medium text-[#666663] dark:text-[#a3a094]">说明</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="rule in headerRuleTypes"
                  :key="rule.type"
                  class="border-b border-[#e5e4df] dark:border-[rgba(227,224,211,0.08)] last:border-0"
                >
                  <td class="px-4 py-3 font-medium text-[#262624] dark:text-[#f1ead8]">
                    <span :class="panelClasses.badgeBlue">{{ rule.name }}</span>
                  </td>
                  <td class="px-4 py-3 text-[#666663] dark:text-[#a3a094]">
                    {{ rule.description }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div :class="[panelClasses.section, 'p-4']">
        <div class="flex items-start gap-3">
          <Info class="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
          <div class="text-sm text-[#666663] dark:text-[#a3a094]">
            <p class="font-medium text-[#262624] dark:text-[#f1ead8]">使用场景</p>
            <ul class="mt-1 space-y-1">
              <li>• 添加额外的认证信息（如 API 版本号）</li>
              <li>• 添加跟踪标记（如请求 ID、来源标识）</li>
              <li>• 删除敏感信息（如用户原始 IP）</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <!-- 代理设置 -->
    <section class="space-y-4">
      <h2 class="text-xl font-semibold text-[#262624] dark:text-[#f1ead8]">
        代理设置
      </h2>

      <div :class="[panelClasses.section, 'p-5']">
        <div class="flex items-center gap-3 mb-4">
          <div class="p-2 rounded-lg bg-blue-500/10">
            <Globe class="h-5 w-5 text-blue-500" />
          </div>
          <h3 class="font-semibold text-[#262624] dark:text-[#f1ead8]">HTTP 代理</h3>
        </div>

        <p class="text-sm text-[#666663] dark:text-[#a3a094] mb-4">
          如果你的服务器无法直接访问某些 API（如在国内访问 OpenAI），可以在端点配置中设置代理。
        </p>

        <div class="space-y-3">
          <div class="p-4 rounded-lg bg-[#f5f5f0]/50 dark:bg-[#1f1d1a]/50">
            <h4 class="font-medium text-[#262624] dark:text-[#f1ead8]">端点级代理</h4>
            <p class="text-sm text-[#666663] dark:text-[#a3a094] mt-1">
              在端点配置中填写代理地址，格式如 <code class="text-xs bg-[#f5f5f0] dark:bg-[#1f1d1a] px-1.5 py-0.5 rounded">http://proxy:8080</code>
            </p>
          </div>

          <div class="p-4 rounded-lg bg-[#f5f5f0]/50 dark:bg-[#1f1d1a]/50">
            <h4 class="font-medium text-[#262624] dark:text-[#f1ead8]">全局代理</h4>
            <p class="text-sm text-[#666663] dark:text-[#a3a094] mt-1">
              也可以通过环境变量 <code class="text-xs bg-[#f5f5f0] dark:bg-[#1f1d1a] px-1.5 py-0.5 rounded">HTTP_PROXY</code> 设置全局代理
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- 系统设置概览 -->
    <section class="space-y-4">
      <h2 class="text-xl font-semibold text-[#262624] dark:text-[#f1ead8]">
        系统设置
      </h2>

      <p class="text-[#666663] dark:text-[#a3a094]">
        在「系统设置」页面可以配置全局参数：
      </p>

      <div class="grid gap-4 md:grid-cols-3">
        <div
          v-for="category in systemSettings"
          :key="category.category"
          :class="[panelClasses.section, 'p-4']"
        >
          <div class="flex items-center gap-2 mb-3">
            <Settings class="h-4 w-4 text-[#cc785c]" />
            <h3 class="font-semibold text-[#262624] dark:text-[#f1ead8]">{{ category.category }}</h3>
          </div>
          <ul class="space-y-2">
            <li
              v-for="item in category.items"
              :key="item.name"
              class="text-sm"
            >
              <span class="font-medium text-[#262624] dark:text-[#f1ead8]">{{ item.name }}</span>
              <span class="text-[#666663] dark:text-[#a3a094]"> - {{ item.description }}</span>
            </li>
          </ul>
        </div>
      </div>
    </section>

    <!-- 健康监控 -->
    <section class="space-y-4">
      <h2 class="text-xl font-semibold text-[#262624] dark:text-[#f1ead8]">
        健康监控
      </h2>

      <div :class="[panelClasses.section, 'p-5']">
        <div class="flex items-center gap-3 mb-4">
          <div class="p-2 rounded-lg bg-green-500/10">
            <Shield class="h-5 w-5 text-green-500" />
          </div>
          <h3 class="font-semibold text-[#262624] dark:text-[#f1ead8]">端点健康检查</h3>
        </div>

        <ul class="space-y-3 text-sm text-[#666663] dark:text-[#a3a094]">
          <li class="flex items-start gap-2">
            <Check class="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span>系统会定期检测端点的可用性</span>
          </li>
          <li class="flex items-start gap-2">
            <Check class="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span>不健康的端点会被自动跳过，请求会路由到其他可用端点</span>
          </li>
          <li class="flex items-start gap-2">
            <Check class="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span>可以在「健康监控」页面查看所有端点的实时状态</span>
          </li>
          <li class="flex items-start gap-2">
            <Check class="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span>支持手动触发健康检查</span>
          </li>
        </ul>
      </div>

      <div :class="[panelClasses.section, 'p-4']">
        <div class="flex items-start gap-3">
          <AlertTriangle class="h-5 w-5 text-yellow-500 flex-shrink-0 mt-0.5" />
          <div class="text-sm text-[#666663] dark:text-[#a3a094]">
            <p class="font-medium text-[#262624] dark:text-[#f1ead8]">端点显示不健康？</p>
            <p class="mt-1">
              检查：1) API URL 是否正确；2) API Key 是否有效；3) 网络是否可达；4) 是否需要配置代理。
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- 下一步 -->
    <section class="pt-4">
      <RouterLink
        to="/guide/faq"
        :class="[panelClasses.section, panelClasses.cardHover, 'p-4 flex items-center gap-3 group']"
      >
        <div class="flex-1">
          <div class="font-medium text-[#262624] dark:text-[#f1ead8]">常见问题</div>
          <div class="text-sm text-[#666663] dark:text-[#a3a094]">查看使用中的常见问题和解答</div>
        </div>
        <ArrowRight class="h-5 w-5 text-[#999] group-hover:text-[#cc785c] transition-colors" />
      </RouterLink>
    </section>
  </div>
</template>
