/**
 * 上游模型获取服务
 *
 * 缓存已移至后端（Redis），前端只保留并发请求去重，避免同时发多个相同请求。
 */
import { ref } from 'vue'
import { adminApi } from '@/api/admin'
import { parseUpstreamModelError } from '@/utils/errorParser'
import type { UpstreamModel } from '@/api/endpoints/types'

export type { UpstreamModel }

type FetchResult = { models: UpstreamModel[]; error?: string; fromCache?: boolean }

// 进行中的请求（用于去重并发请求）
const pendingRequests = new Map<string, Promise<FetchResult>>()

// 请求状态
const loadingMap = ref<Map<string, boolean>>(new Map())

/**
 * 生成请求 key
 */
function getRequestKey(providerId: string, apiKeyId?: string): string {
  return apiKeyId ? `${providerId}:${apiKeyId}` : providerId
}

export function useUpstreamModelsCache() {
  /**
   * 获取上游模型列表
   * @param providerId 提供商ID
   * @param apiKeyId 可选的 API Key ID（用于获取特定 Key 支持的模型）
   * @param forceRefresh 是否强制刷新（跳过后端缓存）
   * @returns 模型列表或错误信息
   */
  async function fetchModels(
    providerId: string,
    apiKeyId?: string,
    forceRefresh = false
  ): Promise<FetchResult> {
    const requestKey = getRequestKey(providerId, apiKeyId)

    // 强制刷新时不复用进行中的请求
    if (!forceRefresh && pendingRequests.has(requestKey)) {
      return pendingRequests.get(requestKey)!
    }

    // 创建新请求
    const requestPromise = (async (): Promise<FetchResult> => {
      try {
        loadingMap.value.set(requestKey, true)
        const response = await adminApi.queryProviderModels(providerId, apiKeyId, forceRefresh)

        if (response.success && response.data?.models) {
          return {
            models: response.data.models,
            // 传递部分格式获取失败的 warning（后端 success=true 但仍可能附带 error）
            error: response.data.error ? parseUpstreamModelError(response.data.error) : undefined,
            fromCache: response.data.from_cache
          }
        } else {
          const rawError = response.data?.error || '获取上游模型失败'
          return { models: [], error: parseUpstreamModelError(rawError) }
        }
      } catch (err: any) {
        const rawError = err.response?.data?.detail || err.message || '获取上游模型失败'
        return { models: [], error: parseUpstreamModelError(rawError) }
      } finally {
        loadingMap.value.set(requestKey, false)
        pendingRequests.delete(requestKey)
      }
    })()

    pendingRequests.set(requestKey, requestPromise)
    return requestPromise
  }

  /**
   * 检查是否正在加载
   */
  function isLoading(providerId: string, apiKeyId?: string): boolean {
    const requestKey = getRequestKey(providerId, apiKeyId)
    return loadingMap.value.get(requestKey) || false
  }

  return {
    fetchModels,
    isLoading,
    loadingMap
  }
}
