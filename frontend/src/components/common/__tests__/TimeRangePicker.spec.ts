import { afterEach, describe, expect, it, vi } from 'vitest'
import { createApp, defineComponent, h, nextTick, ref, type App } from 'vue'

import type { DateRangeParams } from '@/features/usage/types'

// Keep the picker tests focused on the range contract.  The production Select
// implementation is Radix-based and renders its menu in a portal; a small
// passthrough here lets us exercise the native date/datetime inputs directly.
vi.mock('@/components/ui', async () => {
  const { defineComponent, h } = await import('vue')

  const passthrough = (name: string, tag = 'div') => defineComponent({
    name,
    setup(_, { slots }) {
      return () => h(tag, slots.default?.())
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
    },
    emits: ['update:modelValue'],
    setup(props, { attrs, emit }) {
      return () => h('input', {
        ...attrs,
        value: props.modelValue ?? '',
        onInput: (event: Event) => emit(
          'update:modelValue',
          (event.target as HTMLInputElement).value,
        ),
      })
    },
  })

  return {
    Select: passthrough('SelectStub'),
    SelectContent: passthrough('SelectContentStub'),
    SelectItem: passthrough('SelectItemStub'),
    SelectTrigger: passthrough('SelectTriggerStub', 'button'),
    SelectValue: passthrough('SelectValueStub', 'span'),
    Input,
  }
})

import TimeRangePicker from '../TimeRangePicker.vue'
import { createI18n } from '@/i18n'

const mountedApps: Array<{ app: App; root: HTMLElement }> = []

function mountPicker(
  initialValue: DateRangeParams,
  options: { showTime?: boolean } = {},
) {
  const root = document.createElement('div')
  document.body.appendChild(root)
  const value = ref(initialValue)
  const updates: DateRangeParams[] = []

  const Host = defineComponent({
    setup() {
      return () => h(TimeRangePicker, {
        modelValue: value.value,
        showTime: options.showTime,
        showGranularity: false,
        'onUpdate:modelValue': (nextValue: DateRangeParams) => {
          updates.push(nextValue)
          value.value = nextValue
        },
      })
    },
  })

  const app = createApp(Host)
  app.use(createI18n())
  app.mount(root)
  mountedApps.push({ app, root })
  return { root, value, updates }
}

async function setInputValue(input: HTMLInputElement, value: string) {
  input.value = value
  input.dispatchEvent(new Event('input', { bubbles: true }))
  await nextTick()
  await nextTick()
}

afterEach(() => {
  for (const { app, root } of mountedApps.splice(0)) {
    app.unmount()
    root.remove()
  }
  vi.useRealTimers()
})

describe('TimeRangePicker custom bounds', () => {
  it('exposes minute-precision datetime controls and emits both clock fields', async () => {
    const { root, value, updates } = mountPicker({
      preset: 'custom',
      // Date-only values from existing links retain their inclusive-day
      // meaning when the time controls are enabled.
      start_date: '2026-05-06',
      end_date: '2026-05-06',
    }, { showTime: true })

    const inputs = [...root.querySelectorAll<HTMLInputElement>('input')]
    expect(inputs).toHaveLength(2)
    expect(inputs.map(input => input.type)).toEqual(['datetime-local', 'datetime-local'])
    expect(inputs.map(input => input.getAttribute('step'))).toEqual(['60', '60'])
    expect(inputs.map(input => input.value)).toEqual([
      '2026-05-06T00:00',
      '2026-05-06T23:59',
    ])

    await setInputValue(inputs[0], '2026-05-06T09:07')
    await setInputValue(inputs[1], '2026-05-06T18:42')

    expect(value.value).toMatchObject({
      start_date: '2026-05-06T09:07',
      end_date: '2026-05-06T18:42',
    })
    expect(updates.at(-1)).toMatchObject({
      start_date: '2026-05-06T09:07',
      end_date: '2026-05-06T18:42',
    })
  })

  it('keeps date-only controls and values when showTime is disabled', async () => {
    const { root, value, updates } = mountPicker({
      preset: 'custom',
      start_date: '2026-05-06T09:07',
      end_date: '2026-05-06T18:42',
    })

    const inputs = [...root.querySelectorAll<HTMLInputElement>('input')]
    expect(inputs.map(input => input.type)).toEqual(['date', 'date'])
    expect(inputs.map(input => input.getAttribute('step'))).toEqual([null, null])
    expect(inputs.map(input => input.value)).toEqual(['2026-05-06', '2026-05-06'])

    await setInputValue(inputs[1], '2026-05-08')
    await setInputValue(inputs[0], '2026-05-07')

    expect(value.value).toMatchObject({
      start_date: '2026-05-07',
      end_date: '2026-05-08',
    })
    expect(updates.at(-1)).toMatchObject({
      start_date: '2026-05-07',
      end_date: '2026-05-08',
    })
  })

  it('normalizes reversed minute bounds before emitting the custom range', async () => {
    const { root, value, updates } = mountPicker({
      preset: 'custom',
      start_date: '2026-05-06T18:42',
      end_date: '2026-05-06T09:07',
    }, { showTime: true })

    await nextTick()
    await nextTick()
    const inputs = [...root.querySelectorAll<HTMLInputElement>('input')]
    expect(inputs.map(input => input.value)).toEqual([
      '2026-05-06T09:07',
      '2026-05-06T18:42',
    ])
    expect(value.value.start_date).toBe('2026-05-06T09:07')
    expect(value.value.end_date).toBe('2026-05-06T18:42')
    expect(updates.at(-1)).toMatchObject({
      start_date: '2026-05-06T09:07',
      end_date: '2026-05-06T18:42',
    })
  })
})
