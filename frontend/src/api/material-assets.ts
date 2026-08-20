import apiClient from './client'

export type MaterialAssetScope = 'user' | 'admin'

export type MaterialAssetStatus =
  | 'Processing'
  | 'Active'
  | 'Failed'
  | 'processing'
  | 'active'
  | 'failed'

export type MaterialAssetMediaType = 'image' | 'video' | 'audio' | 'file' | 'unknown'
export type ArkMaterialAssetType = 'Image' | 'Video' | 'Audio'
export type ArkCreatableMaterialAssetType = 'Image'

export interface MaterialAssetError {
  code?: string | null
  message?: string | null
}

export interface MaterialAssetGroup {
  id: string
  name: string
  description?: string | null
  group_type: 'AIGC' | 'LivenessFace' | string
  asset_count: number
  created_at?: string | null
  updated_at?: string | null
}

export interface MaterialAsset {
  id: string
  uri?: string | null
  name: string
  status: MaterialAssetStatus | string
  media_type?: MaterialAssetMediaType | string | null
  asset_type?: ArkMaterialAssetType | string | null
  mime_type?: string | null
  group_id?: string | null
  group_name?: string | null
  source_type?: 'url' | 'upload' | 'generated' | string | null
  size_bytes?: number | null
  width?: number | null
  height?: number | null
  duration_seconds?: number | null
  user_id?: string | null
  username?: string | null
  provider_id?: string | null
  provider_name?: string | null
  requires_real_person_verification?: boolean
  verification_status?: string | null
  error?: MaterialAssetError | null
  error_code?: string | null
  error_message?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface MaterialAssetGroupListResponse {
  items: MaterialAssetGroup[]
  total: number
}

export interface MaterialAssetListResponse {
  items: MaterialAsset[]
  total: number
  page: number
  page_size: number
  pages?: number
}

export interface MaterialAssetListParams {
  group_id?: string
  search?: string
  type?: string
  status?: string
  user_id?: string
  page?: number
  page_size?: number
}

export interface MaterialAssetGroupListParams {
  search?: string
  user_id?: string
}

export interface CreateMaterialAssetGroupRequest {
  name: string
  description?: string
  group_type?: 'AIGC'
  user_id?: string
}

export interface CreateMaterialAssetFromUrlRequest {
  url: string
  name?: string
  group_id: string
  asset_type: ArkCreatableMaterialAssetType
  user_id?: string
}

export interface RenameMaterialAssetRequest {
  name: string
}

export interface MaterialAssetVerificationSession {
  id: string
  status: string
  verification_url?: string | null
  h5_link?: string | null
  group_id?: string | null
  error_message?: string | null
  expires_at?: string | number | null
}

export interface CreateMaterialAssetVerificationSessionRequest {
  callback_url: string
  user_id?: string
}

function materialAssetsBasePath(scope: MaterialAssetScope): string {
  return scope === 'admin' ? '/api/admin/material-assets' : '/api/material-assets'
}

function requestForScope<T extends { user_id?: string }>(
  scope: MaterialAssetScope,
  request: T,
): T {
  if (scope === 'admin' || request.user_id === undefined) return request
  const scopedRequest = { ...request }
  delete scopedRequest.user_id
  return scopedRequest
}

function adminOwnerParams(
  scope: MaterialAssetScope,
  ownerUserId?: string,
): { user_id: string } | undefined {
  const userId = ownerUserId?.trim()
  return scope === 'admin' && userId ? { user_id: userId } : undefined
}

export function createMaterialAssetsApi(scope: MaterialAssetScope) {
  const basePath = materialAssetsBasePath(scope)

  return {
    async listGroups(params: MaterialAssetGroupListParams = {}): Promise<MaterialAssetGroupListResponse> {
      const response = await apiClient.get<MaterialAssetGroupListResponse>(`${basePath}/groups`, {
        params: requestForScope(scope, params),
      })
      return response.data
    },

    async createGroup(payload: CreateMaterialAssetGroupRequest): Promise<MaterialAssetGroup> {
      const response = await apiClient.post<MaterialAssetGroup>(
        `${basePath}/groups`,
        requestForScope(scope, payload),
      )
      return response.data
    },

    async getGroup(groupId: string, ownerUserId?: string): Promise<MaterialAssetGroup> {
      const response = await apiClient.get<MaterialAssetGroup>(
        `${basePath}/groups/${encodeURIComponent(groupId)}`,
        { params: adminOwnerParams(scope, ownerUserId) },
      )
      return response.data
    },

    async renameGroup(
      groupId: string,
      name: string,
      ownerUserId?: string,
    ): Promise<MaterialAssetGroup> {
      const response = await apiClient.patch<MaterialAssetGroup>(
        `${basePath}/groups/${encodeURIComponent(groupId)}`,
        { name },
        { params: adminOwnerParams(scope, ownerUserId) },
      )
      return response.data
    },

    async deleteGroup(groupId: string, ownerUserId?: string): Promise<void> {
      await apiClient.delete(`${basePath}/groups/${encodeURIComponent(groupId)}`, {
        params: adminOwnerParams(scope, ownerUserId),
      })
    },

    async listAssets(params: MaterialAssetListParams = {}): Promise<MaterialAssetListResponse> {
      const response = await apiClient.get<MaterialAssetListResponse>(`${basePath}/assets`, {
        params: requestForScope(scope, params),
      })
      return response.data
    },

    async getAsset(assetId: string, ownerUserId?: string): Promise<MaterialAsset> {
      const response = await apiClient.get<MaterialAsset>(
        `${basePath}/assets/${encodeURIComponent(assetId)}`,
        { params: adminOwnerParams(scope, ownerUserId) },
      )
      return response.data
    },

    async createFromUrl(payload: CreateMaterialAssetFromUrlRequest): Promise<MaterialAsset> {
      const response = await apiClient.post<MaterialAsset>(
        `${basePath}/assets/url`,
        requestForScope(scope, payload),
      )
      return response.data
    },

    async renameAsset(
      assetId: string,
      payload: RenameMaterialAssetRequest,
      ownerUserId?: string,
    ): Promise<MaterialAsset> {
      const response = await apiClient.patch<MaterialAsset>(
        `${basePath}/assets/${encodeURIComponent(assetId)}`,
        payload,
        { params: adminOwnerParams(scope, ownerUserId) },
      )
      return response.data
    },

    async deleteAsset(assetId: string, ownerUserId?: string): Promise<void> {
      await apiClient.delete(`${basePath}/assets/${encodeURIComponent(assetId)}`, {
        params: adminOwnerParams(scope, ownerUserId),
      })
    },

    async createVerificationSession(
      payload: CreateMaterialAssetVerificationSessionRequest,
    ): Promise<MaterialAssetVerificationSession> {
      const response = await apiClient.post<MaterialAssetVerificationSession>(
        `${basePath}/verification-sessions`,
        requestForScope(scope, payload),
      )
      return response.data
    },

    async getVerificationSession(
      sessionId: string,
      ownerUserId?: string,
    ): Promise<MaterialAssetVerificationSession> {
      const response = await apiClient.get<MaterialAssetVerificationSession>(
        `${basePath}/verification-sessions/${encodeURIComponent(sessionId)}`,
        { params: adminOwnerParams(scope, ownerUserId) },
      )
      return response.data
    },

    async getPreviewBlob(
      assetId: string,
      signal?: AbortSignal,
      ownerUserId?: string,
    ): Promise<Blob> {
      const response = await apiClient.get<Blob>(
        `${basePath}/assets/${encodeURIComponent(assetId)}/preview`,
        {
          responseType: 'blob',
          signal,
          params: adminOwnerParams(scope, ownerUserId),
        },
      )
      return response.data
    },
  }
}
