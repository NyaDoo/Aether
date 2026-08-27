import { describe, expect, it } from 'vitest'

import { resolveUsageOperationPresentation } from '../operation'

describe('resolveUsageOperationPresentation', () => {
  it.each([
    ['video.create', '生成视频'],
    ['video.remix', '重混视频'],
    ['video.cancel', '取消视频'],
    ['video.delete', '删除视频'],
    ['asset_library.create_group', '创建素材组'],
    ['asset_library.list_groups', '列出素材组'],
    ['asset_library.get_group', '获取素材组'],
    ['asset_library.update_group', '更新素材组'],
    ['asset_library.delete_group', '删除素材组'],
    ['asset_library.create_asset', '创建素材'],
    ['asset_library.list_assets', '列出素材'],
    ['asset_library.get_asset', '获取素材'],
    ['asset_library.update_asset', '更新素材'],
    ['asset_library.delete_asset', '删除素材'],
    ['asset_library.create_visual_validation', '创建视觉校验'],
    ['asset_library.get_visual_validation', '获取视觉校验'],
  ])('formats canonical operation %s as %s', (operation, label) => {
    const presentation = resolveUsageOperationPresentation(operation)

    expect(presentation).toMatchObject({
      canonical: operation,
      label,
    })
    expect(presentation?.ariaLabel).toBe(`请求操作：${label}`)
  })

  it.each([
    ['video_create', 'video.create'],
    ['video_remix', 'video.remix'],
    ['video_cancel', 'video.cancel'],
    ['video_delete', 'video.delete'],
    ['asset_group_create', 'asset_library.create_group'],
    ['asset_group_list', 'asset_library.list_groups'],
    ['asset_group_get', 'asset_library.get_group'],
    ['asset_group_update', 'asset_library.update_group'],
    ['asset_group_delete', 'asset_library.delete_group'],
    ['asset_create', 'asset_library.create_asset'],
    ['asset_list', 'asset_library.list_assets'],
    ['asset_get', 'asset_library.get_asset'],
    ['asset_update', 'asset_library.update_asset'],
    ['asset_delete', 'asset_library.delete_asset'],
    ['visual_validate_create', 'asset_library.create_visual_validation'],
    ['visual_validate_get', 'asset_library.get_visual_validation'],
  ])('normalizes legacy alias %s to %s', (operation, canonical) => {
    expect(resolveUsageOperationPresentation(operation)?.canonical).toBe(canonical)
  })

  it.each([null, undefined, '', 'chat.completions'])('does not render unsupported operation %s', (operation) => {
    expect(resolveUsageOperationPresentation(operation)).toBeNull()
  })
})
