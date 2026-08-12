import type { VideoTaskItem } from '@/api/video-tasks'

type VideoTaskModelFields = Pick<
  VideoTaskItem,
  'model' | 'global_model_name' | 'mapped_model' | 'observed_model'
>

export interface VideoTaskModelIdentity {
  /** Stable public/global identity. Falls back to the exact request model for older records. */
  primary: string
  /** Exact request model, only surfaced separately when it differs from the primary identity. */
  requested: string | null
  /** Provider-facing target selected by routing/model mapping. */
  mapped: string | null
  /** Model/version reported by the upstream provider. */
  observed: string | null
}

function normalizedModel(value: string | null | undefined): string | null {
  const normalized = value?.trim()
  return normalized || null
}

/**
 * Keeps model identities directional and explicit:
 * request/global identity, mapping target, then upstream observation.
 */
export function videoTaskModelIdentity(task: VideoTaskModelFields): VideoTaskModelIdentity {
  const requestModel = normalizedModel(task.model)
  const primary = normalizedModel(task.global_model_name) ?? requestModel ?? '-'

  return {
    primary,
    requested: requestModel && requestModel !== primary ? requestModel : null,
    mapped: normalizedModel(task.mapped_model),
    observed: normalizedModel(task.observed_model),
  }
}
