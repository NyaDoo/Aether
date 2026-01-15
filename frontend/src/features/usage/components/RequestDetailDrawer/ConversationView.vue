<template>
  <div class="conversation-view">
    <!-- 解析错误提示 -->
    <div
      v-if="conversation.parseError"
      class="parse-error"
    >
      <AlertCircle class="w-4 h-4" />
      <span>{{ conversation.parseError }}</span>
    </div>

    <!-- 空内容提示 -->
    <div
      v-else-if="!conversation.system && conversation.messages.length === 0"
      class="text-sm text-muted-foreground"
    >
      {{ emptyMessage }}
    </div>

    <template v-else>
      <!-- System Prompt -->
      <div
        v-if="conversation.system"
        class="message-block system"
      >
        <div class="message-header">
          <Settings class="w-3.5 h-3.5" />
          <span>System</span>
        </div>
        <div class="message-content">
          <pre class="whitespace-pre-wrap">{{ conversation.system }}</pre>
        </div>
      </div>

      <!-- 对话消息 -->
      <div
        v-for="(msg, index) in conversation.messages"
        :key="index"
        class="message-block"
        :class="[msg.role, msg.type]"
      >
        <div class="message-header">
          <component
            :is="getRoleIcon(msg.role)"
            class="w-3.5 h-3.5"
          />
          <span>{{ getRoleLabel(msg.role) }}</span>
          <Badge
            v-if="msg.type === 'thinking'"
            variant="secondary"
            class="ml-2 text-xs"
          >
            思考
          </Badge>
          <Badge
            v-if="msg.type === 'tool_use'"
            variant="outline"
            class="ml-2 text-xs"
          >
            {{ msg.metadata?.toolName || '工具调用' }}
          </Badge>
          <Badge
            v-if="msg.type === 'tool_result'"
            variant="outline"
            class="ml-2 text-xs"
          >
            工具结果
          </Badge>
          <Badge
            v-if="msg.type === 'image'"
            variant="secondary"
            class="ml-2 text-xs"
          >
            图片
          </Badge>
        </div>

        <div class="message-content">
          <!-- 思考过程：可折叠 -->
          <details
            v-if="msg.type === 'thinking'"
            class="thinking-details"
          >
            <summary class="thinking-summary">
              <ChevronRight class="w-4 h-4 chevron" />
              <span class="text-muted-foreground">点击展开思考过程 ({{ msg.content.length }} 字符)</span>
            </summary>
            <pre class="thinking-content whitespace-pre-wrap">{{ msg.content }}</pre>
          </details>

          <!-- 工具调用/结果：代码块样式 -->
          <pre
            v-else-if="msg.type === 'tool_use' || msg.type === 'tool_result'"
            class="tool-content whitespace-pre-wrap"
          >{{ msg.content }}</pre>

          <!-- 普通文本 -->
          <pre
            v-else
            class="whitespace-pre-wrap"
          >{{ msg.content }}</pre>
        </div>
      </div>

      <!-- 流式响应标记 -->
      <div
        v-if="conversation.isStream"
        class="stream-indicator"
      >
        <Zap class="w-3 h-3" />
        <span>流式响应</span>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { User, Bot, Settings, Wrench, AlertCircle, ChevronRight, Zap } from 'lucide-vue-next'
import Badge from '@/components/ui/badge.vue'
import type { ExtractedConversation, MessageRole } from '../../utils/messageExtractor'

defineProps<{
  conversation: ExtractedConversation
  emptyMessage: string
}>()

const getRoleIcon = (role: MessageRole) => {
  switch (role) {
    case 'user':
      return User
    case 'assistant':
      return Bot
    case 'system':
      return Settings
    case 'tool':
      return Wrench
    default:
      return User
  }
}

const getRoleLabel = (role: MessageRole) => {
  switch (role) {
    case 'user':
      return 'User'
    case 'assistant':
      return 'Assistant'
    case 'system':
      return 'System'
    case 'tool':
      return 'Tool'
    default:
      return role
  }
}
</script>

<style scoped>
.conversation-view {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 500px;
  overflow-y: auto;
  padding: 4px;
}

.parse-error {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  background: hsl(var(--destructive) / 0.1);
  border: 1px solid hsl(var(--destructive) / 0.3);
  border-radius: 8px;
  color: hsl(var(--destructive));
  font-size: 13px;
}

.message-block {
  border-radius: 8px;
  overflow: hidden;
}

.message-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 500;
}

.message-content {
  padding: 0 12px 12px;
  font-size: 13px;
  line-height: 1.6;
}

.message-content pre {
  margin: 0;
  font-family: inherit;
}

/* User 消息样式 */
.message-block.user {
  background: hsl(var(--primary) / 0.08);
  border: 1px solid hsl(var(--primary) / 0.2);
}

.message-block.user .message-header {
  color: hsl(var(--primary));
}

/* Assistant 消息样式 */
.message-block.assistant {
  background: hsl(var(--muted) / 0.5);
  border: 1px solid hsl(var(--border));
}

.message-block.assistant .message-header {
  color: hsl(var(--foreground));
}

/* System 消息样式 */
.message-block.system {
  background: hsl(var(--muted) / 0.3);
  border: 1px dashed hsl(var(--border));
}

.message-block.system .message-header {
  color: hsl(var(--muted-foreground));
}

/* Tool 消息样式 */
.message-block.tool {
  background: hsl(var(--muted) / 0.3);
  border: 1px solid hsl(var(--border));
}

/* 思考过程样式 */
.message-block.thinking {
  background: hsl(var(--muted) / 0.3);
  border: 1px solid hsl(var(--border));
}

.thinking-details {
  cursor: pointer;
}

.thinking-summary {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 0;
  list-style: none;
  user-select: none;
}

.thinking-summary::-webkit-details-marker {
  display: none;
}

.thinking-summary .chevron {
  transition: transform 0.2s;
}

.thinking-details[open] .chevron {
  transform: rotate(90deg);
}

.thinking-content {
  margin-top: 8px;
  padding: 12px;
  background: hsl(var(--muted) / 0.5);
  border-radius: 6px;
  font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 12px;
  max-height: 300px;
  overflow-y: auto;
}

/* 工具调用样式 */
.tool-content {
  padding: 12px;
  background: hsl(var(--muted) / 0.5);
  border-radius: 6px;
  font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 12px;
  max-height: 200px;
  overflow-y: auto;
}

/* 流式响应标记 */
.stream-indicator {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  font-size: 11px;
  color: hsl(var(--muted-foreground));
  background: hsl(var(--muted) / 0.3);
  border-radius: 4px;
  width: fit-content;
}
</style>
