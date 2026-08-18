import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createApp, nextTick, type App, type Component } from 'vue'
import KeyFormDialog from '@/features/providers/components/KeyFormDialog.vue'
import OAuthKeyEditDialog from '@/features/providers/components/OAuthKeyEditDialog.vue'
import type { EndpointAPIKey } from '@/api/endpoints'

const endpointMocks = vi.hoisted(() => ({
  addProviderKey: vi.fn(),
  updateProviderKey: vi.fn(),
  getAllCapabilities: vi.fn(),
  sortApiFormats: vi.fn((formats: string[]) => [...formats].sort()),
}))

vi.mock('@/api/endpoints', () => ({
  addProviderKey: endpointMocks.addProviderKey,
  updateProviderKey: endpointMocks.updateProviderKey,
  getAllCapabilities: endpointMocks.getAllCapabilities,
  sortApiFormats: endpointMocks.sortApiFormats,
  API_FORMATS: {
    DOUBAO_VIDEO: 'doubao:video',
    DOUBAO_ASSET_LIBRARY: 'doubao:asset_library',
  },
}))

vi.mock('@/components/ui', async () => {
  const { defineComponent, h, inject, provide } = await import('vue')
  const SelectContextKey = Symbol('SelectContext')

  const passthrough = (name: string, tag = 'div') => defineComponent({
    name,
    setup(_, { slots }) {
      return () => h(tag, slots.default?.())
    },
  })

  const Dialog = defineComponent({
    name: 'DialogStub',
    props: {
      modelValue: Boolean,
    },
    setup(props, { slots }) {
      return () => props.modelValue
        ? h('section', [slots.default?.(), slots.footer?.()])
        : null
    },
  })

  const Input = defineComponent({
    name: 'InputStub',
    inheritAttrs: false,
    props: {
      modelValue: {
        type: [String, Number],
        default: '',
      },
      masked: Boolean,
    },
    emits: ['update:modelValue'],
    setup(props, { attrs, emit }) {
      return () => h('input', {
        ...attrs,
        value: props.modelValue ?? '',
        onInput: (event: Event) => emit('update:modelValue', (event.target as HTMLInputElement).value),
      })
    },
  })

  const Label = defineComponent({
    name: 'LabelStub',
    inheritAttrs: false,
    props: {
      for: String,
    },
    setup(props, { attrs, slots }) {
      return () => h('label', { ...attrs, for: props.for }, slots.default?.())
    },
  })

  const Button = defineComponent({
    name: 'ButtonStub',
    inheritAttrs: false,
    props: {
      disabled: Boolean,
      variant: String,
    },
    setup(props, { attrs, slots }) {
      return () => h('button', {
        ...attrs,
        disabled: props.disabled,
        type: attrs.type ?? 'button',
      }, slots.default?.())
    },
  })

  const Switch = defineComponent({
    name: 'SwitchStub',
    inheritAttrs: false,
    props: {
      modelValue: Boolean,
    },
    emits: ['update:modelValue'],
    setup(props, { attrs, emit }) {
      return () => h('input', {
        ...attrs,
        type: 'checkbox',
        checked: props.modelValue,
        onChange: (event: Event) => emit('update:modelValue', (event.target as HTMLInputElement).checked),
      })
    },
  })

  const Select = defineComponent({
    name: 'SelectStub',
    inheritAttrs: false,
    props: {
      modelValue: String,
    },
    emits: ['update:modelValue'],
    setup(props, { attrs, emit, slots }) {
      provide(SelectContextKey, {
        select: (value: string) => emit('update:modelValue', value),
        modelValue: props.modelValue,
      })

      return () => h('div', {
        ...attrs,
        'data-select': 'true',
        'data-value': props.modelValue,
      }, slots.default?.())
    },
  })

  const SelectItem = defineComponent({
    name: 'SelectItemStub',
    inheritAttrs: false,
    props: {
      value: {
        type: String,
        required: true,
      },
    },
    setup(props, { attrs, slots }) {
      const context = inject<{ select: (value: string) => void } | null>(SelectContextKey, null)
      return () => h('button', {
        ...attrs,
        type: 'button',
        'data-select-item': props.value,
        onClick: () => context?.select(props.value),
      }, slots.default?.())
    },
  })

  return {
    Dialog,
    Button,
    Input,
    Label,
    Switch,
    Select,
    SelectTrigger: passthrough('SelectTriggerStub'),
    SelectValue: passthrough('SelectValueStub', 'span'),
    SelectContent: passthrough('SelectContentStub'),
    SelectItem,
  }
})

vi.mock('@/components/common/JsonImportInput.vue', async () => {
  const { defineComponent, h } = await import('vue')

  return {
    default: defineComponent({
      name: 'JsonImportInputStub',
      props: {
        modelValue: {
          type: String,
          default: '',
        },
      },
      emits: ['update:modelValue'],
      setup(props, { emit }) {
        return () => h('textarea', {
          value: props.modelValue,
          onInput: (event: Event) => emit('update:modelValue', (event.target as HTMLTextAreaElement).value),
        })
      },
    }),
  }
})

vi.mock('@/composables/useToast', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
  }),
}))

vi.mock('@/composables/useConfirm', () => ({
  useConfirm: () => ({
    confirmWarning: vi.fn().mockResolvedValue(true),
  }),
}))

vi.mock('lucide-vue-next', async () => {
  const { defineComponent, h } = await import('vue')
  const Icon = defineComponent({
    name: 'IconStub',
    setup() {
      return () => h('span')
    },
  })

  return {
    CircleHelp: Icon,
    Key: Icon,
    SquarePen: Icon,
  }
})

const mountedApps: Array<{ app: App, root: HTMLElement }> = []

function createProviderKey(overrides: Partial<EndpointAPIKey> = {}): EndpointAPIKey {
  return {
    id: 'provider-key-1',
    provider_id: 'provider-1',
    api_formats: ['openai:chat'],
    api_key_masked: 'sk-***',
    auth_type: 'api_key',
    name: 'Primary key',
    rate_multipliers: null,
    internal_priority: 10,
    rpm_limit: 30,
    concurrent_limit: null,
    allowed_models: null,
    capabilities: null,
    cache_ttl_minutes: 5,
    max_probe_interval_minutes: 32,
    health_score: 100,
    consecutive_failures: 0,
    request_count: 0,
    success_count: 0,
    error_count: 0,
    success_rate: 1,
    avg_response_time_ms: 0,
    is_active: true,
    note: '',
    created_at: '2026-04-27T00:00:00Z',
    updated_at: '2026-04-27T00:00:00Z',
    auto_fetch_models: false,
    model_include_patterns: [],
    model_exclude_patterns: [],
    ...overrides,
  }
}

function mountDialog(component: Component, props: Record<string, unknown>) {
  const root = document.createElement('div')
  document.body.appendChild(root)
  const app = createApp(component, props)
  app.mount(root)
  mountedApps.push({ app, root })
  return root
}

async function settle() {
  await nextTick()
  await Promise.resolve()
  await nextTick()
}

function findInput(root: HTMLElement, id: string) {
  const input = root.querySelector<HTMLInputElement>(`#${id}`)
  expect(input).not.toBeNull()
  return input as HTMLInputElement
}

function updateInput(input: HTMLInputElement, value: string) {
  input.value = value
  input.dispatchEvent(new Event('input', { bubbles: true }))
}

function updateTextarea(textarea: HTMLTextAreaElement, value: string) {
  textarea.value = value
  textarea.dispatchEvent(new Event('input', { bubbles: true }))
}

async function submit(root: HTMLElement) {
  const form = root.querySelector('form')
  expect(form).not.toBeNull()
  form?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
  await settle()
}

function lastUpdatePayload() {
  const calls = endpointMocks.updateProviderKey.mock.calls
  expect(calls.length).toBeGreaterThan(0)
  return calls[calls.length - 1][1] as Record<string, unknown>
}

beforeEach(() => {
  endpointMocks.addProviderKey.mockReset()
  endpointMocks.updateProviderKey.mockReset()
  endpointMocks.getAllCapabilities.mockReset()
  endpointMocks.sortApiFormats.mockClear()

  endpointMocks.addProviderKey.mockResolvedValue(createProviderKey())
  endpointMocks.updateProviderKey.mockResolvedValue(createProviderKey())
  endpointMocks.getAllCapabilities.mockResolvedValue([])
})

afterEach(() => {
  for (const { app, root } of mountedApps.splice(0)) {
    app.unmount()
    root.remove()
  }
})

describe('provider key concurrent_limit form behavior', () => {
  it('creates a Volcengine AK/SK key restricted to the Ark asset-library format', async () => {
    const root = mountDialog(KeyFormDialog, {
      open: true,
      endpoint: null,
      editingKey: null,
      providerId: 'provider-volcengine',
      providerType: 'custom',
      availableApiFormats: ['doubao:video', 'doubao:asset_library'],
    })
    await settle()

    expect(root.querySelector('[data-select]')?.getAttribute('data-value')).toBe('api_key')
    expect(root.querySelector('[data-provider-auth-layout]')?.getAttribute('data-provider-auth-layout')).toBe('inline')
    const akSkOption = root.querySelector<HTMLButtonElement>('[data-select-item="volc_aksk"]')
    expect(akSkOption).not.toBeNull()
    akSkOption?.click()
    await settle()

    expect(root.querySelector('[data-provider-auth-layout]')?.getAttribute('data-provider-auth-layout')).toBe('stacked')

    const nameInput = root.querySelector<HTMLInputElement>('input[placeholder="例如：主 Key、备用 Key 1"]')
    const accessKeyInput = root.querySelector<HTMLInputElement>('input[id^="volc-access-key-id-"]')
    const secretKeyInput = root.querySelector<HTMLInputElement>('input[id^="volc-secret-access-key-"]')
    const securityTokenInput = root.querySelector<HTMLInputElement>('input[id^="volc-security-token-"]')
    expect(nameInput).not.toBeNull()
    expect(accessKeyInput).not.toBeNull()
    expect(secretKeyInput).not.toBeNull()
    expect(securityTokenInput).not.toBeNull()
    expect(root.querySelector('input[id^="ark-account-id-"]')).toBeNull()
    expect(root.querySelector('input[id^="ark-project-"]')).toBeNull()

    updateInput(nameInput as HTMLInputElement, 'Ark asset signer')
    updateInput(accessKeyInput as HTMLInputElement, 'AKLT-example')
    updateInput(secretKeyInput as HTMLInputElement, 'secret-example')
    updateInput(securityTokenInput as HTMLInputElement, 'session-token')
    await submit(root)

    expect(endpointMocks.addProviderKey).toHaveBeenCalledWith(
      'provider-volcengine',
      expect.objectContaining({
        auth_type: 'volc_aksk',
        api_key: '',
        api_formats: ['doubao:asset_library'],
        auth_config: {
          access_key_id: 'AKLT-example',
          secret_access_key: 'secret-example',
          security_token: 'session-token',
          region: 'cn-beijing',
          service: 'ark',
        },
      }),
    )
  })

  it('preserves an existing Volcengine AK/SK secret when credential fields stay blank', async () => {
    const root = mountDialog(KeyFormDialog, {
      open: true,
      endpoint: null,
      editingKey: createProviderKey({
        auth_type: 'volc_aksk',
        api_formats: ['doubao:asset_library'],
        api_key_masked: '[Volcengine AK/SK]',
      }),
      providerId: 'provider-volcengine',
      providerType: 'custom',
      availableApiFormats: ['doubao:asset_library'],
    })
    await settle()

    expect(root.querySelector('input[id^="ark-account-id-"]')).toBeNull()
    expect(root.querySelector('input[id^="ark-project-"]')).toBeNull()
    expect(root.textContent).toContain('AK 与 SK 留空表示保持原凭据')

    await submit(root)

    const payload = lastUpdatePayload()
    expect(payload.auth_type).toBe('volc_aksk')
    expect(payload.api_formats).toEqual(['doubao:asset_library'])
    expect(payload).not.toHaveProperty('api_key')
    expect(payload).not.toHaveProperty('auth_config')
  })

  it('creates Bearer credentials without Ark account or project binding', async () => {
    const root = mountDialog(KeyFormDialog, {
      open: true,
      endpoint: null,
      editingKey: null,
      providerId: 'provider-relay',
      providerType: 'custom',
      availableApiFormats: ['doubao:asset_library', 'doubao:video'],
    })
    await settle()

    const bearerOption = root.querySelector<HTMLButtonElement>('[data-select-item="bearer"]')
    expect(bearerOption).not.toBeNull()
    bearerOption?.click()
    await settle()

    expect(root.querySelector('[data-provider-auth-layout]')?.getAttribute('data-provider-auth-layout')).toBe('inline')

    const nameInput = root.querySelector<HTMLInputElement>('input[placeholder="例如：主 Key、备用 Key 1"]')
    const secretInput = root.querySelector<HTMLInputElement>('input[id^="api-key-"]')
    expect(nameInput).not.toBeNull()
    expect(secretInput).not.toBeNull()
    expect(root.querySelector('input[id^="ark-account-id-"]')).toBeNull()
    expect(root.querySelector('input[id^="ark-project-"]')).toBeNull()
    updateInput(nameInput as HTMLInputElement, 'Relay bearer')
    updateInput(secretInput as HTMLInputElement, 'relay-token')
    await submit(root)

    expect(endpointMocks.addProviderKey).toHaveBeenCalledWith(
      'provider-relay',
      expect.objectContaining({
        auth_type: 'bearer',
        api_key: 'relay-token',
      }),
    )
    expect(endpointMocks.addProviderKey.mock.calls[0][1].auth_config).toBeUndefined()
  })

  it('selects the upstream header for an Ark asset-library API Key', async () => {
    const root = mountDialog(KeyFormDialog, {
      open: true,
      endpoint: null,
      editingKey: null,
      providerId: 'provider-asset-relay',
      providerType: 'custom',
      availableApiFormats: ['doubao:asset_library'],
    })
    await settle()

    const headerSelect = root.querySelector<HTMLElement>('[data-ark-api-key-header]')
    expect(headerSelect?.getAttribute('data-value')).toBe('x-api-key')
    const apiKeyHeaderOption = headerSelect?.querySelector<HTMLButtonElement>('[data-select-item="api-key"]')
    expect(apiKeyHeaderOption).not.toBeNull()
    apiKeyHeaderOption?.click()
    await settle()

    const nameInput = root.querySelector<HTMLInputElement>('input[placeholder="例如：主 Key、备用 Key 1"]')
    const secretInput = root.querySelector<HTMLInputElement>('input[id^="api-key-"]')
    updateInput(nameInput as HTMLInputElement, 'Ark API Key relay')
    updateInput(secretInput as HTMLInputElement, 'asset-api-key')
    await submit(root)

    expect(endpointMocks.addProviderKey).toHaveBeenCalledWith(
      'provider-asset-relay',
      expect.objectContaining({
        auth_type: 'api_key',
        api_key: 'asset-api-key',
        api_formats: ['doubao:asset_library'],
        auth_config: {
          api_key_header: 'api-key',
        },
      }),
    )
  })

  it('shows the upstream API Key header when only the Ark format overrides Bearer auth', async () => {
    const root = mountDialog(KeyFormDialog, {
      open: true,
      endpoint: null,
      editingKey: createProviderKey({
        auth_type: 'bearer',
        auth_type_by_format: {
          'doubao:asset_library': 'api_key',
        },
        api_formats: ['doubao:asset_library'],
      }),
      providerId: 'provider-asset-relay',
      providerType: 'custom',
      availableApiFormats: ['doubao:asset_library'],
    })
    await settle()

    const headerSelect = root.querySelector<HTMLElement>('[data-ark-api-key-header]')
    expect(headerSelect).not.toBeNull()
    const apiKeyHeaderOption = headerSelect?.querySelector<HTMLButtonElement>('[data-select-item="api-key"]')
    expect(apiKeyHeaderOption).not.toBeNull()
    apiKeyHeaderOption?.click()
    await submit(root)

    const payload = lastUpdatePayload()
    expect(payload.auth_type).toBe('bearer')
    expect(payload.auth_type_by_format).toEqual({
      'doubao:asset_library': 'api_key',
    })
    expect(payload.auth_config).toEqual({
      api_key_header: 'api-key',
    })
  })

  it('creates API Key credentials for Doubao video without Ark binding', async () => {
    const root = mountDialog(KeyFormDialog, {
      open: true,
      endpoint: null,
      editingKey: null,
      providerId: 'provider-video-relay',
      providerType: 'custom',
      availableApiFormats: ['doubao:video'],
    })
    await settle()

    const nameInput = root.querySelector<HTMLInputElement>('input[placeholder="例如：主 Key、备用 Key 1"]')
    const secretInput = root.querySelector<HTMLInputElement>('input[id^="api-key-"]')
    expect(root.querySelector('input[id^="ark-account-id-"]')).toBeNull()
    expect(root.querySelector('input[id^="ark-project-"]')).toBeNull()

    updateInput(nameInput as HTMLInputElement, 'Doubao video relay')
    updateInput(secretInput as HTMLInputElement, 'video-api-key')
    await submit(root)

    expect(endpointMocks.addProviderKey).toHaveBeenCalledWith(
      'provider-video-relay',
      expect.objectContaining({
        auth_type: 'api_key',
        api_key: 'video-api-key',
        api_formats: ['doubao:video'],
      }),
    )
    expect(endpointMocks.addProviderKey.mock.calls[0][1].auth_config).toBeUndefined()
  })

  it('does not expose or submit Ark account and project fields when editing', async () => {
    const root = mountDialog(KeyFormDialog, {
      open: true,
      endpoint: null,
      editingKey: createProviderKey({
        auth_type: 'bearer',
        api_formats: ['doubao:video'],
        api_key_masked: 'token-***',
      }),
      providerId: 'provider-video-relay',
      providerType: 'custom',
      availableApiFormats: ['doubao:video'],
    })
    await settle()

    expect(root.querySelector('input[id^="ark-account-id-"]')).toBeNull()
    expect(root.querySelector('input[id^="ark-project-"]')).toBeNull()
    await submit(root)

    const payload = lastUpdatePayload()
    expect(payload).not.toHaveProperty('api_key')
    expect(payload).not.toHaveProperty('auth_config')
  })

  it.each([
    ['api_key', 'bearer'],
    ['bearer', 'api_key'],
  ] as const)(
    'switches %s to %s without overwriting hidden Ark auth_config',
    async (currentAuthType, nextAuthType) => {
      const root = mountDialog(KeyFormDialog, {
        open: true,
        endpoint: null,
        editingKey: createProviderKey({
          auth_type: currentAuthType,
          api_formats: ['doubao:asset_library'],
          api_key_masked: 'relay-***',
        }),
        providerId: 'provider-asset-relay',
        providerType: 'custom',
        availableApiFormats: ['doubao:asset_library'],
      })
      await settle()

      const nextAuthTypeOption = root.querySelector<HTMLButtonElement>(`[data-select-item="${nextAuthType}"]`)
      expect(nextAuthTypeOption).not.toBeNull()
      nextAuthTypeOption?.click()
      await settle()

      expect(root.querySelector('input[id^="ark-account-id-"]')).toBeNull()
      expect(root.querySelector('input[id^="ark-project-"]')).toBeNull()
      if (nextAuthType === 'api_key') {
        expect(root.querySelector('[data-ark-api-key-header]')?.getAttribute('data-value')).toBe('')
      }

      await submit(root)

      const payload = lastUpdatePayload()
      expect(payload.auth_type).toBe(nextAuthType)
      expect(payload).not.toHaveProperty('api_key')
      expect(payload).not.toHaveProperty('auth_config')
    },
  )

  it('uses the existing flat section style and responsive grids', async () => {
    const root = mountDialog(KeyFormDialog, {
      open: true,
      endpoint: null,
      editingKey: null,
      providerId: 'provider-volcengine',
      providerType: 'custom',
      availableApiFormats: ['doubao:asset_library'],
    })
    await settle()

    const sectionNames = Array.from(
      root.querySelectorAll<HTMLElement>('[data-provider-key-section]'),
      section => section.dataset.providerKeySection,
    )
    expect(sectionNames).toEqual([
      'basic',
      'authentication',
      'api-formats',
      'advanced-authentication',
      'scheduling',
    ])
    expect(root.querySelector('[data-provider-scheduling-grid]')?.className).toContain('sm:grid-cols-2')
    expect(root.querySelector('[data-provider-scheduling-grid]')?.className).toContain('xl:grid-cols-5')
    expect(root.innerHTML).not.toContain('bg-amber-500/[0.04]')
    expect(root.innerHTML).not.toContain('bg-sky-500/[0.04]')
  })

  it('lets Vertex AI keys switch to Service Account JSON and submit auth_config', async () => {
    const root = mountDialog(KeyFormDialog, {
      open: true,
      endpoint: null,
      editingKey: null,
      providerId: 'provider-vertex',
      providerType: 'vertex_ai',
      availableApiFormats: ['gemini:generate_content', 'claude:messages'],
    })
    await settle()

    const serviceAccountOption = root.querySelector<HTMLButtonElement>('[data-select-item="service_account"]')
    expect(serviceAccountOption).not.toBeNull()
    serviceAccountOption?.click()
    await settle()

    const nameInput = root.querySelector<HTMLInputElement>('input[placeholder="例如：主 Key、备用 Key 1"]')
    expect(nameInput).not.toBeNull()
    updateInput(nameInput as HTMLInputElement, 'Vertex service account')

    const textarea = root.querySelector<HTMLTextAreaElement>('textarea')
    expect(textarea).not.toBeNull()
    updateTextarea(textarea as HTMLTextAreaElement, JSON.stringify({
      client_email: 'svc@example.iam.gserviceaccount.com',
      private_key: '-----BEGIN PRIVATE KEY-----\\nTEST\\n-----END PRIVATE KEY-----\\n',
      project_id: 'demo-project',
    }))

    await submit(root)

    expect(endpointMocks.addProviderKey).toHaveBeenCalledWith('provider-vertex', expect.objectContaining({
      auth_type: 'service_account',
      auth_config: expect.objectContaining({
        client_email: 'svc@example.iam.gserviceaccount.com',
        private_key: '-----BEGIN PRIVATE KEY-----\\nTEST\\n-----END PRIVATE KEY-----\\n',
        project_id: 'demo-project',
      }),
      api_formats: ['gemini:generate_content'],
    }))
  })

  it('keeps Gemini embedding selectable for Vertex AI keys', async () => {
    const root = mountDialog(KeyFormDialog, {
      open: true,
      endpoint: null,
      editingKey: null,
      providerId: 'provider-vertex',
      providerType: 'vertex_ai',
      availableApiFormats: ['gemini:generate_content', 'gemini:embedding', 'claude:messages'],
    })
    await settle()

    expect(root.textContent).toContain('Gemini Embedding')

    const serviceAccountOption = root.querySelector<HTMLButtonElement>('[data-select-item="service_account"]')
    expect(serviceAccountOption).not.toBeNull()
    serviceAccountOption?.click()
    await settle()

    expect(root.textContent).toContain('Gemini Embedding')
  })

  it('hydrates and serializes a positive concurrent_limit number from the normal key form', async () => {
    const root = mountDialog(KeyFormDialog, {
      open: true,
      endpoint: null,
      editingKey: createProviderKey({ rpm_limit: 42, concurrent_limit: 3 }),
      providerId: 'provider-1',
      providerType: 'openai',
      availableApiFormats: ['openai:chat'],
    })
    await settle()

    const concurrentLimitInput = findInput(root, 'concurrent_limit')
    expect(concurrentLimitInput.value).toBe('3')
    expect(findInput(root, 'rpm_limit').value).toBe('42')

    updateInput(concurrentLimitInput, '5')
    await submit(root)

    const payload = lastUpdatePayload()
    expect(payload.concurrent_limit).toBe(5)
    expect(typeof payload.concurrent_limit).toBe('number')
    expect(payload.concurrent_limit).not.toBe('')
    expect(payload.rpm_limit).toBe(42)
  })

  it('serializes cleared normal key concurrent_limit as null instead of an empty string', async () => {
    const root = mountDialog(KeyFormDialog, {
      open: true,
      endpoint: null,
      editingKey: createProviderKey({ rpm_limit: 24, concurrent_limit: 6 }),
      providerId: 'provider-1',
      providerType: 'openai',
      availableApiFormats: ['openai:chat'],
    })
    await settle()

    updateInput(findInput(root, 'concurrent_limit'), '')
    await submit(root)

    const payload = lastUpdatePayload()
    expect(payload).toHaveProperty('concurrent_limit', null)
    expect(payload.concurrent_limit).not.toBe('')
    expect(payload.rpm_limit).toBe(24)
  })

  it('hydrates and serializes a positive concurrent_limit number from the OAuth edit form', async () => {
    const root = mountDialog(OAuthKeyEditDialog, {
      open: true,
      editingKey: createProviderKey({
        id: 'oauth-key-1',
        auth_type: 'oauth',
        name: 'OAuth account',
        rpm_limit: 35,
        concurrent_limit: 3,
      }),
    })
    await settle()

    const concurrentLimitInput = findInput(root, 'concurrent_limit')
    expect(concurrentLimitInput.value).toBe('3')
    expect(findInput(root, 'rpm_limit').value).toBe('35')

    updateInput(concurrentLimitInput, '7')
    await submit(root)

    const payload = lastUpdatePayload()
    expect(endpointMocks.updateProviderKey).toHaveBeenCalledWith('oauth-key-1', expect.any(Object))
    expect(payload.concurrent_limit).toBe(7)
    expect(typeof payload.concurrent_limit).toBe('number')
    expect(payload.concurrent_limit).not.toBe('')
    expect(payload.rpm_limit).toBe(35)
  })

  it('serializes cleared OAuth concurrent_limit as null instead of an empty string', async () => {
    const root = mountDialog(OAuthKeyEditDialog, {
      open: true,
      editingKey: createProviderKey({
        id: 'oauth-key-2',
        auth_type: 'oauth',
        rpm_limit: 18,
        concurrent_limit: 4,
      }),
    })
    await settle()

    updateInput(findInput(root, 'concurrent_limit'), '')
    await submit(root)

    const payload = lastUpdatePayload()
    expect(payload).toHaveProperty('concurrent_limit', null)
    expect(payload.concurrent_limit).not.toBe('')
    expect(payload.rpm_limit).toBe(18)
  })

  it('keeps zero concurrent_limit as a numeric unlimited value', async () => {
    const root = mountDialog(OAuthKeyEditDialog, {
      open: true,
      editingKey: createProviderKey({
        id: 'oauth-key-zero',
        auth_type: 'oauth',
        rpm_limit: 11,
        concurrent_limit: 2,
      }),
    })
    await settle()

    updateInput(findInput(root, 'concurrent_limit'), '0')
    await submit(root)

    const payload = lastUpdatePayload()
    expect(payload.concurrent_limit).toBe(0)
    expect(typeof payload.concurrent_limit).toBe('number')
    expect(payload.rpm_limit).toBe(11)
  })
})
