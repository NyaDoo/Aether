export type UsageOperationCategory = 'video' | 'asset-group' | 'asset' | 'visual-validation'

export interface UsageOperationPresentation {
  canonical: string
  label: string
  category: UsageOperationCategory
  className: string
  title: string
  ariaLabel: string
}

interface UsageOperationDefinition {
  label: string
  category: UsageOperationCategory
  className: string
}

const VIDEO_CLASS = 'border-violet-500/30 bg-violet-500/5 text-violet-700 dark:text-violet-300'
const ASSET_GROUP_CLASS = 'border-cyan-500/30 bg-cyan-500/5 text-cyan-700 dark:text-cyan-300'
const ASSET_CLASS = 'border-emerald-500/30 bg-emerald-500/5 text-emerald-700 dark:text-emerald-300'
const VISUAL_VALIDATION_CLASS = 'border-amber-500/30 bg-amber-500/5 text-amber-700 dark:text-amber-300'

const OPERATION_DEFINITIONS: Record<string, UsageOperationDefinition> = {
  'video.create': {
    label: '生成视频',
    category: 'video',
    className: VIDEO_CLASS,
  },
  'video.remix': {
    label: '重混视频',
    category: 'video',
    className: VIDEO_CLASS,
  },
  'video.cancel': {
    label: '取消视频',
    category: 'video',
    className: VIDEO_CLASS,
  },
  'video.delete': {
    label: '删除视频',
    category: 'video',
    className: VIDEO_CLASS,
  },
  'asset_library.create_group': {
    label: '创建素材组',
    category: 'asset-group',
    className: ASSET_GROUP_CLASS,
  },
  'asset_library.list_groups': {
    label: '列出素材组',
    category: 'asset-group',
    className: ASSET_GROUP_CLASS,
  },
  'asset_library.get_group': {
    label: '获取素材组',
    category: 'asset-group',
    className: ASSET_GROUP_CLASS,
  },
  'asset_library.update_group': {
    label: '更新素材组',
    category: 'asset-group',
    className: ASSET_GROUP_CLASS,
  },
  'asset_library.delete_group': {
    label: '删除素材组',
    category: 'asset-group',
    className: ASSET_GROUP_CLASS,
  },
  'asset_library.create_asset': {
    label: '创建素材',
    category: 'asset',
    className: ASSET_CLASS,
  },
  'asset_library.list_assets': {
    label: '列出素材',
    category: 'asset',
    className: ASSET_CLASS,
  },
  'asset_library.get_asset': {
    label: '获取素材',
    category: 'asset',
    className: ASSET_CLASS,
  },
  'asset_library.update_asset': {
    label: '更新素材',
    category: 'asset',
    className: ASSET_CLASS,
  },
  'asset_library.delete_asset': {
    label: '删除素材',
    category: 'asset',
    className: ASSET_CLASS,
  },
  'asset_library.create_visual_validation': {
    label: '创建视觉校验',
    category: 'visual-validation',
    className: VISUAL_VALIDATION_CLASS,
  },
  'asset_library.get_visual_validation': {
    label: '获取视觉校验',
    category: 'visual-validation',
    className: VISUAL_VALIDATION_CLASS,
  },
}

const OPERATION_ALIASES: Record<string, string> = {
  video_create: 'video.create',
  video_remix: 'video.remix',
  video_cancel: 'video.cancel',
  video_delete: 'video.delete',
  asset_group_create: 'asset_library.create_group',
  asset_group_list: 'asset_library.list_groups',
  asset_group_get: 'asset_library.get_group',
  asset_group_update: 'asset_library.update_group',
  asset_group_delete: 'asset_library.delete_group',
  asset_create: 'asset_library.create_asset',
  asset_list: 'asset_library.list_assets',
  asset_get: 'asset_library.get_asset',
  asset_update: 'asset_library.update_asset',
  asset_delete: 'asset_library.delete_asset',
  visual_validate_create: 'asset_library.create_visual_validation',
  visual_validate_get: 'asset_library.get_visual_validation',
  visual_validation_create: 'asset_library.create_visual_validation',
  visual_validation_get: 'asset_library.get_visual_validation',
}

export function resolveUsageOperationPresentation(
  operation: string | null | undefined,
): UsageOperationPresentation | null {
  const normalized = normalizeOperation(operation)
  if (!normalized) return null

  const canonical = OPERATION_ALIASES[normalized] ?? normalized
  const definition = OPERATION_DEFINITIONS[canonical]
  if (!definition) return null

  return {
    canonical,
    ...definition,
    title: `请求操作：${definition.label} (${canonical})`,
    ariaLabel: `请求操作：${definition.label}`,
  }
}

function normalizeOperation(operation: string | null | undefined): string | null {
  if (typeof operation !== 'string') return null
  const normalized = operation.trim().toLowerCase().replace(/[\s-]+/g, '_')
  return normalized || null
}
