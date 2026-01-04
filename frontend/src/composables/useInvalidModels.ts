import { computed, type Ref, type ComputedRef } from 'vue'

/**
 * 检测失效模型的 composable
 *
 * 用于检测 allowed_models 中已不存在于 globalModels 的模型名称，
 * 这些模型可能已被删除但引用未清理。
 *
 * @example
 * ```typescript
 * const { invalidModels } = useInvalidModels(
 *   computed(() => form.value.allowed_models),
 *   globalModels
 * )
 * ```
 */
export interface ModelWithName {
  name: string
}

export function useInvalidModels<T extends ModelWithName>(
  allowedModels: Ref<string[]> | ComputedRef<string[]>,
  globalModels: Ref<T[]>
): { invalidModels: ComputedRef<string[]> } {
  const validModelNames = computed(() =>
    new Set(globalModels.value.map(m => m.name))
  )

  const invalidModels = computed(() =>
    allowedModels.value.filter(name => !validModelNames.value.has(name))
  )

  return { invalidModels }
}
