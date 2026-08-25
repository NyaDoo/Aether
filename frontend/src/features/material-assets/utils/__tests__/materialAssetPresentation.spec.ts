import { describe, expect, it } from 'vitest'

import {
  buildMaterialAssetVideoReference,
  materialAssetErrorMessage,
  materialAssetMediaType,
  materialAssetOfficialUrl,
  materialAssetUri,
  normalizeMaterialAssetStatus,
} from '@/features/material-assets/utils/materialAssetPresentation'

describe('material asset presentation', () => {
  it('normalizes upstream lifecycle states and prefers the official asset ID and URL', () => {
    expect(normalizeMaterialAssetStatus('Succeeded')).toBe('active')
    expect(normalizeMaterialAssetStatus('Rejected')).toBe('failed')
    expect(normalizeMaterialAssetStatus('Processing')).toBe('processing')
    expect(materialAssetUri({ id: 'asset-ark-1', uri: 'asset://legacy-local-id' })).toBe('asset://asset-ark-1')
    expect(materialAssetUri({ id: 'asset-ark-1' })).toBe('asset://asset-ark-1')
    expect(materialAssetOfficialUrl({
      url: 'https://ark.example.test/asset-ark-1?signature=short-lived',
      uri: 'asset://asset-ark-1',
    })).toBe('https://ark.example.test/asset-ark-1?signature=short-lived')
    expect(materialAssetOfficialUrl({
      url: null,
      uri: 'https://legacy.example.test/asset-ark-1',
    })).toBe('https://legacy.example.test/asset-ark-1')
    expect(materialAssetOfficialUrl({ url: null, uri: 'asset://asset-ark-1' })).toBeNull()
  })

  it('builds the official Seedance content object for each supported media type', () => {
    expect(materialAssetMediaType({ asset_type: 'Video' })).toBe('video')
    expect(buildMaterialAssetVideoReference({
      id: 'image-1',
      media_type: 'image',
    })).toEqual({
      type: 'image_url',
      image_url: { url: 'asset://image-1' },
      role: 'reference_image',
    })
    expect(buildMaterialAssetVideoReference({
      id: 'video-1',
      uri: 'asset://upstream-video',
      media_type: 'video',
    })).toEqual({
      type: 'video_url',
      video_url: { url: 'asset://video-1' },
      role: 'reference_video',
    })
    expect(buildMaterialAssetVideoReference({
      id: 'audio-1',
      media_type: 'audio',
    })).toEqual({
      type: 'audio_url',
      audio_url: { url: 'asset://audio-1' },
      role: 'reference_audio',
    })
    expect(buildMaterialAssetVideoReference({ id: 'file-1', media_type: 'file' })).toBeNull()
    expect(buildMaterialAssetVideoReference({
      id: 'official-audio-1',
      asset_type: 'Audio',
    })).toEqual({
      type: 'audio_url',
      audio_url: { url: 'asset://official-audio-1' },
      role: 'reference_audio',
    })
  })

  it('maps known upstream error codes without exposing opaque messages', () => {
    expect(materialAssetErrorMessage({
      id: 'asset-1',
      name: 'person',
      status: 'Failed',
      media_type: 'image',
      error_code: 'REAL_PERSON_VERIFICATION_REQUIRED',
    })).toContain('身份验证')
  })
})
