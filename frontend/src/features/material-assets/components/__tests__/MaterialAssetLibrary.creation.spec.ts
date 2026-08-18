import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const source = readFileSync(
  resolve(process.cwd(), 'src/features/material-assets/components/MaterialAssetLibrary.vue'),
  'utf8',
)

describe('MaterialAssetLibrary admin group creation', () => {
  it('keeps the create action available before an admin owner filter is applied', () => {
    const groupHeader = source
      .split('aria-label="创建素材组"')[1]
      ?.split('</Button>')[0]

    expect(groupHeader).toBeTruthy()
    expect(groupHeader).toContain(':disabled="!canMutate"')
    expect(groupHeader).toContain('@click="openCreateGroupDialog"')
    expect(groupHeader).not.toContain(':disabled="!canCreate"')
  })

  it('collects the required owner in the create dialog and applies it after creation', () => {
    expect(source).toContain('id="material-group-owner"')
    expect(source).toContain('v-model="newGroupOwnerUserId"')
    expect(source).toContain('(isAdmin && !newGroupOwnerUserId.trim())')
    expect(source).toContain('user_id: targetUserId || undefined')
    expect(source).toContain('appliedAdminUserId.value = targetUserId')
  })
})

describe('MaterialAssetLibrary real-person verification', () => {
  it('sends the current material library page as the official CallbackURL', () => {
    const startVerification = source
      .split('async function startVerification()')[1]
      ?.split('function reopenVerificationSession()')[0]

    expect(startVerification).toBeTruthy()
    expect(startVerification).toContain('user_id: ownerAtStart || undefined')
    expect(startVerification).not.toContain('return_url')
    expect(startVerification).toContain('const callbackUrl = new URL(window.location.href)')
    expect(startVerification).toContain("callbackUrl.hash = ''")
    expect(startVerification).toContain('callback_url: callbackUrl.toString()')
  })
})

describe('MaterialAssetLibrary K23 URL asset contract', () => {
  it('creates only HTTPS image assets', () => {
    const createAsset = source
      .split('async function createAssetFromUrl()')[1]
      ?.split('function openRenameDialog')[0]

    expect(source).toContain('title="通过公网 URL 创建图片素材"')
    expect(source).not.toContain('v-model="sourceUrlAssetType"')
    expect(createAsset).toContain("parsed.protocol !== 'https:'")
    expect(createAsset).toContain("asset_type: 'Image'")
  })
})
