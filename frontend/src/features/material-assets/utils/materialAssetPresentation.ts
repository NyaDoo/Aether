import type { MaterialAsset } from '@/api/material-assets'

export type NormalizedMaterialAssetStatus = 'processing' | 'active' | 'failed'

const ERROR_MESSAGES: Record<string, string> = {
  FACE_VERIFICATION_REQUIRED: '该真人素材需要完成身份验证后才能使用',
  REAL_PERSON_VERIFICATION_REQUIRED: '该真人素材需要完成身份验证后才能使用',
  INVALID_ASSET_URL: '素材 URL 无效或无法访问',
  UNSUPPORTED_MEDIA_TYPE: '不支持该素材格式',
  ASSET_PROCESSING_FAILED: '素材处理失败，请检查源文件后重试',
  ASSET_NOT_FOUND: '素材不存在或已被删除',
  PERMISSION_DENIED: '没有权限访问该素材',
  QUOTA_EXCEEDED: '素材额度不足，请清理后重试',
}

export function normalizeMaterialAssetStatus(status: string | null | undefined): NormalizedMaterialAssetStatus {
  const normalized = status?.trim().toLowerCase()
  if (normalized === 'active' || normalized === 'ready' || normalized === 'succeeded') return 'active'
  if (normalized === 'failed' || normalized === 'error' || normalized === 'rejected') return 'failed'
  return 'processing'
}

export function materialAssetStatusLabel(status: string | null | undefined): string {
  switch (normalizeMaterialAssetStatus(status)) {
    case 'active':
      return 'Active'
    case 'failed':
      return 'Failed'
    default:
      return 'Processing'
  }
}

export function materialAssetUri(asset: Pick<MaterialAsset, 'id' | 'uri'>): string {
  const explicit = asset.uri?.trim()
  return explicit || `asset://${asset.id}`
}

export function materialAssetErrorMessage(asset: MaterialAsset): string | null {
  const code = asset.error?.code?.trim() || asset.error_code?.trim()
  const message = asset.error?.message?.trim() || asset.error_message?.trim()
  if (code && ERROR_MESSAGES[code]) return ERROR_MESSAGES[code]
  return message || (normalizeMaterialAssetStatus(asset.status) === 'failed' ? '素材处理失败' : null)
}

export function materialAssetRequiresVerification(asset: MaterialAsset): boolean {
  if (asset.requires_real_person_verification) return true
  const code = asset.error?.code?.trim() || asset.error_code?.trim()
  return code === 'FACE_VERIFICATION_REQUIRED' || code === 'REAL_PERSON_VERIFICATION_REQUIRED'
}

export function materialAssetMediaLabel(mediaType: string | null | undefined): string {
  switch (mediaType?.trim().toLowerCase()) {
    case 'image':
      return '图片'
    case 'video':
      return '视频'
    case 'audio':
      return '音频'
    default:
      return '文件'
  }
}

export function materialAssetMediaType(
  asset: Pick<MaterialAsset, 'media_type' | 'asset_type'>,
): string {
  return asset.media_type?.trim().toLowerCase()
    || asset.asset_type?.trim().toLowerCase()
    || 'unknown'
}

export function materialAssetSupportsVideoReference(
  asset: Pick<MaterialAsset, 'media_type' | 'asset_type'>,
): boolean {
  return ['image', 'video', 'audio'].includes(materialAssetMediaType(asset))
}

export function buildMaterialAssetVideoReference(
  asset: Pick<MaterialAsset, 'id' | 'uri' | 'media_type' | 'asset_type'>,
): Record<string, unknown> | null {
  const mediaType = materialAssetMediaType(asset)
  if (!mediaType || !materialAssetSupportsVideoReference(asset)) return null

  const field = mediaType === 'video' ? 'video_url' : mediaType === 'audio' ? 'audio_url' : 'image_url'
  const role = mediaType === 'video' ? 'reference_video' : mediaType === 'audio' ? 'reference_audio' : 'reference_image'
  return {
    type: field,
    [field]: { url: materialAssetUri(asset) },
    role,
  }
}
