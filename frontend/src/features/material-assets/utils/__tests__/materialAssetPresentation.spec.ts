import { describe, expect, it } from 'vitest'

import {
  buildMaterialAssetVideoReference,
  materialAssetMediaType,
  materialAssetErrorMessage,
  materialAssetUri,
  normalizeMaterialAssetStatus,
} from '@/features/material-assets/utils/materialAssetPresentation'

describe('material asset presentation', () => {
  it('normalizes upstream lifecycle states and preserves an explicit asset URI', () => {
    expect(normalizeMaterialAssetStatus('Succeeded')).toBe('active')
    expect(normalizeMaterialAssetStatus('Rejected')).toBe('failed')
    expect(normalizeMaterialAssetStatus('Processing')).toBe('processing')
    expect(materialAssetUri({ id: 'local-1', uri: 'asset://ark-1' })).toBe('asset://ark-1')
    expect(materialAssetUri({ id: 'local-1' })).toBe('asset://local-1')
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
      video_url: { url: 'asset://upstream-video' },
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
