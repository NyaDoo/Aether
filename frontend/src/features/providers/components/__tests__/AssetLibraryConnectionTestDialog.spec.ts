import { describe, expect, it, vi } from 'vitest'
import { createApp, defineComponent, h } from 'vue'

import { createI18n } from '@/i18n'

vi.mock('@/components/ui', async () => {
  const { defineComponent, h } = await import('vue')
  const passthrough = (name: string, tag = 'div') => defineComponent({
    name,
    setup(_, { slots }) {
      return () => h(tag, slots.default?.())
    },
  })
  const DialogStub = defineComponent({
    name: 'DialogStub',
    props: {
      title: { type: String, default: '' },
      description: { type: String, default: '' },
    },
    setup(props, { slots }) {
      return () => h('div', [props.title, props.description, slots.default?.(), slots.footer?.()])
    },
  })

  return {
    Button: passthrough('ButtonStub', 'button'),
    Dialog: DialogStub,
    Label: passthrough('LabelStub', 'label'),
    Select: passthrough('SelectStub'),
    SelectContent: passthrough('SelectContentStub'),
    SelectItem: passthrough('SelectItemStub'),
    SelectTrigger: passthrough('SelectTriggerStub'),
    SelectValue: passthrough('SelectValueStub'),
  }
})

vi.mock('lucide-vue-next', async () => {
  const { defineComponent, h } = await import('vue')
  const Icon = defineComponent({
    name: 'IconStub',
    setup() {
      return () => h('span')
    },
  })
  return { Activity: Icon, Loader2: Icon }
})

import AssetLibraryConnectionTestDialog from '@/features/providers/components/AssetLibraryConnectionTestDialog.vue'

describe('AssetLibraryConnectionTestDialog', () => {
  it('distinguishes exact endpoints without exposing URL credentials or query values', () => {
    const root = document.createElement('div')
    document.body.appendChild(root)
    const app = createApp(defineComponent({
      setup() {
        return () => h(AssetLibraryConnectionTestDialog, {
          open: true,
          keyName: 'Relay key',
          selectedEndpointId: 'endpoint-123456789',
          endpoints: [{
            id: 'endpoint-123456789',
            provider_id: 'provider-1',
            provider_name: 'Relay',
            api_format: 'doubao:asset_library',
            base_url: 'https://user:password@relay.example.test/base?token=secret',
            custom_path: '/seedance/assets/?signature=private',
            max_retries: 0,
            is_active: true,
            total_keys: 1,
            active_keys: 1,
            created_at: '2026-01-01T00:00:00Z',
            updated_at: '2026-01-01T00:00:00Z',
          }],
        })
      },
    }))
    app.use(createI18n())
    app.mount(root)

    expect(root.textContent).toContain('素材库基础连通性测试')
    expect(root.textContent).toContain('只验证基础连通性')
    expect(root.textContent).toContain('ListAssetGroups')
    expect(root.textContent).toContain('https://relay.example.test/seedance/assets/ · endpoint')
    expect(root.textContent).not.toContain('user')
    expect(root.textContent).not.toContain('password')
    expect(root.textContent).not.toContain('signature')
    expect(root.textContent).not.toContain('private')

    app.unmount()
    root.remove()
  })
})
