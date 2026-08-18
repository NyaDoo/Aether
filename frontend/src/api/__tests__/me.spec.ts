import { beforeEach, describe, expect, it, vi } from 'vitest'

const { patchMock, postMock } = vi.hoisted(() => ({
  patchMock: vi.fn(),
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  default: {
    patch: patchMock,
    post: postMock,
  },
}))

import { meApi } from '@/api/me'

describe('meApi API key status', () => {
  beforeEach(() => {
    patchMock.mockReset()
    postMock.mockReset()
    patchMock.mockResolvedValue({
      data: {
        id: 'user-key-1',
        is_active: false,
      },
    })
  })

  it('sends the selected credential type when creating Volcengine AK/SK', async () => {
    postMock.mockResolvedValue({
      data: {
        id: 'user-aksk-1',
        name: 'ark signer',
        key_display: '',
        credential_type: 'volc_aksk',
        access_key_id: 'AKLTEXAMPLE',
        secret_access_key: 'secret-example',
        is_active: true,
        is_locked: false,
      },
    })

    const created = await meApi.createApiKey({
      name: 'ark signer',
      credential_type: 'volc_aksk',
    })

    expect(postMock).toHaveBeenCalledWith('/api/users/me/api-keys', {
      name: 'ark signer',
      credential_type: 'volc_aksk',
    })
    expect(created.access_key_id).toBe('AKLTEXAMPLE')
    expect(created.secret_access_key).toBe('secret-example')
  })

  it('preserves a standard API key create request without requiring a credential type', async () => {
    postMock.mockResolvedValue({
      data: {
        id: 'user-key-2',
        name: 'standard',
        key: 'sk-standard-live',
        key_display: 'sk-standard...live',
        credential_type: 'api_key',
        is_active: true,
        is_locked: false,
      },
    })

    const created = await meApi.createApiKey({ name: 'standard' })

    expect(postMock).toHaveBeenCalledWith('/api/users/me/api-keys', { name: 'standard' })
    expect(created.key).toBe('sk-standard-live')
  })

  it('sends the desired disabled state in the patch body', async () => {
    await meApi.toggleApiKey('user-key-1', false)

    expect(patchMock).toHaveBeenCalledWith(
      '/api/users/me/api-keys/user-key-1',
      { is_active: false },
    )
  })
})
