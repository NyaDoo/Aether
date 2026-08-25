import { beforeEach, describe, expect, it, vi } from 'vitest'

const { getMock, postMock, patchMock, deleteMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  patchMock: vi.fn(),
  deleteMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  default: {
    get: getMock,
    post: postMock,
    patch: patchMock,
    delete: deleteMock,
  },
}))

import { createMaterialAssetsApi } from '@/api/material-assets'

describe('material assets API', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    patchMock.mockReset()
    deleteMock.mockReset()
  })

  it('keeps user and admin resources on separate authenticated API surfaces', async () => {
    getMock.mockResolvedValue({ data: { items: [], total: 0, page: 1, page_size: 20 } })

    await createMaterialAssetsApi('user').listAssets({
      status: 'Processing',
      user_id: 'other-user',
    })
    await createMaterialAssetsApi('admin').listAssets({ user_id: 'user-1' })

    expect(getMock).toHaveBeenNthCalledWith(1, '/api/material-assets/assets', {
      params: { status: 'Processing' },
    })
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/admin/material-assets/assets', {
      params: { user_id: 'user-1' },
    })
  })

  it('drops user impersonation fields on user writes and preserves explicit admin owners', async () => {
    const group = {
      id: 'group-1',
      name: '参考素材',
      group_type: 'AIGC',
      asset_count: 0,
    }
    postMock.mockResolvedValue({ data: group })

    await createMaterialAssetsApi('user').createGroup({
      name: '参考素材',
      user_id: 'other-user',
    })
    await createMaterialAssetsApi('admin').createGroup({
      name: '参考素材',
      user_id: 'owner-user',
    })

    expect(postMock).toHaveBeenNthCalledWith(1, '/api/material-assets/groups', {
      name: '参考素材',
    })
    expect(postMock).toHaveBeenNthCalledWith(2, '/api/admin/material-assets/groups', {
      name: '参考素材',
      user_id: 'owner-user',
    })
  })

  it('allows the official Ark group type to be omitted without injecting a value', async () => {
    const group = {
      id: 'group-1',
      name: '角色素材',
      description: '视频生成参考图',
      group_type: 'AIGC',
      asset_count: 0,
    }
    postMock.mockResolvedValue({ data: group })

    await expect(createMaterialAssetsApi('user').createGroup({
      name: '角色素材',
      description: '视频生成参考图',
    })).resolves.toEqual(group)

    expect(postMock).toHaveBeenCalledWith('/api/material-assets/groups', {
      name: '角色素材',
      description: '视频生成参考图',
    })
  })

  it('loads previews as blobs through the authenticated client', async () => {
    const blob = new Blob(['image'], { type: 'image/png' })
    const controller = new AbortController()
    getMock.mockResolvedValue({ data: blob })

    await expect(
      createMaterialAssetsApi('admin').getPreviewBlob('asset/id', controller.signal),
    ).resolves.toBe(blob)

    expect(getMock).toHaveBeenCalledWith(
      '/api/admin/material-assets/assets/asset%2Fid/preview',
      { responseType: 'blob', signal: controller.signal, params: undefined },
    )

    await createMaterialAssetsApi('admin').getPreviewBlob(
      'asset/id',
      controller.signal,
      'owner-1',
      '/api/admin/material-assets/assets/asset%2Fid/preview',
    )
    expect(getMock).toHaveBeenLastCalledWith(
      '/api/admin/material-assets/assets/asset%2Fid/preview',
      { responseType: 'blob', signal: controller.signal, params: { user_id: 'owner-1' } },
    )
  })

  it('passes every official Ark URL asset type without narrowing it to images', async () => {
    const asset = {
      id: 'asset-1',
      name: 'reference.png',
      status: 'Processing',
      media_type: 'image',
    }
    postMock.mockResolvedValue({ data: asset })
    const api = createMaterialAssetsApi('user')

    for (const assetType of ['Image', 'Video', 'Audio'] as const) {
      await api.createFromUrl({
        url: `https://example.test/reference-${assetType.toLowerCase()}`,
        group_id: 'group-official-1',
        asset_type: assetType,
      })
      expect(postMock).toHaveBeenLastCalledWith('/api/material-assets/assets/url', {
        url: `https://example.test/reference-${assetType.toLowerCase()}`,
        group_id: 'group-official-1',
        asset_type: assetType,
      })
    }
  })

  it('uses the official callback_url field for real-person verification', async () => {
    postMock.mockResolvedValue({
      data: {
        id: 'session-1',
        status: 'Pending',
        h5_link: 'https://verify.example.test/session-1',
      },
    })

    await createMaterialAssetsApi('user').createVerificationSession({
      callback_url: 'https://aether.example.test/api/material-assets/verification-callback',
      user_id: 'other-user',
    })

    expect(postMock).toHaveBeenCalledWith('/api/material-assets/verification-sessions', {
      callback_url: 'https://aether.example.test/api/material-assets/verification-callback',
    })
  })

  it('polls a real-person verification session by its session id', async () => {
    const session = {
      id: 'session/1',
      status: 'Succeeded',
      group_id: 'liveness-group-1',
    }
    getMock.mockResolvedValue({ data: session })

    await expect(
      createMaterialAssetsApi('admin').getVerificationSession('session/1'),
    ).resolves.toEqual(session)

    expect(getMock).toHaveBeenCalledWith(
      '/api/admin/material-assets/verification-sessions/session%2F1',
      { params: undefined },
    )
  })

  it('carries the applied owner through admin reads, previews, and mutations', async () => {
    const group = { id: 'group-1', name: 'Owner group', group_type: 'AIGC', asset_count: 0 }
    const asset = { id: 'asset-1', name: 'Owner asset', status: 'Active' }
    const session = { id: 'session-1', status: 'Pending' }
    getMock.mockImplementation(async (path: string) => {
      if (path.includes('/groups/')) return { data: group }
      if (path.includes('/verification-sessions/')) return { data: session }
      if (path.endsWith('/preview')) return { data: new Blob(['preview']) }
      return { data: asset }
    })
    patchMock.mockResolvedValue({ data: asset })
    deleteMock.mockResolvedValue({})

    const api = createMaterialAssetsApi('admin')
    await api.getGroup('group-1', 'owner-42')
    await api.renameGroup('group-1', 'Renamed', 'owner-42')
    await api.deleteGroup('group-1', 'owner-42')
    await api.getAsset('asset-1', 'owner-42')
    await api.renameAsset('asset-1', { name: 'Renamed asset' }, 'owner-42')
    await api.deleteAsset('asset-1', 'owner-42')
    await api.getVerificationSession('session-1', 'owner-42')
    await api.getPreviewBlob('asset-1', undefined, 'owner-42')

    expect(getMock).toHaveBeenCalledWith('/api/admin/material-assets/groups/group-1', {
      params: { user_id: 'owner-42' },
    })
    expect(patchMock).toHaveBeenNthCalledWith(
      1,
      '/api/admin/material-assets/groups/group-1',
      { name: 'Renamed' },
      { params: { user_id: 'owner-42' } },
    )
    expect(deleteMock).toHaveBeenNthCalledWith(
      1,
      '/api/admin/material-assets/groups/group-1',
      { params: { user_id: 'owner-42' } },
    )
    expect(getMock).toHaveBeenCalledWith('/api/admin/material-assets/assets/asset-1', {
      params: { user_id: 'owner-42' },
    })
    expect(patchMock).toHaveBeenNthCalledWith(
      2,
      '/api/admin/material-assets/assets/asset-1',
      { name: 'Renamed asset' },
      { params: { user_id: 'owner-42' } },
    )
    expect(deleteMock).toHaveBeenNthCalledWith(
      2,
      '/api/admin/material-assets/assets/asset-1',
      { params: { user_id: 'owner-42' } },
    )
    expect(getMock).toHaveBeenCalledWith('/api/admin/material-assets/verification-sessions/session-1', {
      params: { user_id: 'owner-42' },
    })
    expect(getMock).toHaveBeenCalledWith('/api/admin/material-assets/assets/asset-1/preview', {
      responseType: 'blob',
      signal: undefined,
      params: { user_id: 'owner-42' },
    })
  })

  it('ignores an owner hint on user-scope reads and mutations', async () => {
    getMock.mockResolvedValue({ data: { id: 'asset-1', name: 'Asset', status: 'Active' } })
    patchMock.mockResolvedValue({ data: { id: 'asset-1', name: 'Renamed', status: 'Active' } })
    deleteMock.mockResolvedValue({})

    const api = createMaterialAssetsApi('user')
    await api.getAsset('asset-1', 'other-owner')
    await api.renameAsset('asset-1', { name: 'Renamed' }, 'other-owner')
    await api.deleteAsset('asset-1', 'other-owner')

    expect(getMock).toHaveBeenCalledWith('/api/material-assets/assets/asset-1', {
      params: undefined,
    })
    expect(patchMock).toHaveBeenCalledWith(
      '/api/material-assets/assets/asset-1',
      { name: 'Renamed' },
      { params: undefined },
    )
    expect(deleteMock).toHaveBeenCalledWith('/api/material-assets/assets/asset-1', {
      params: undefined,
    })
  })
})
