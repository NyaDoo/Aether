import { describe, expect, it } from 'vitest'

import { videoTaskModelIdentity } from '../videoTaskModelIdentity'

describe('videoTaskModelIdentity', () => {
  it('keeps global, mapped, and observed model identities distinct', () => {
    expect(videoTaskModelIdentity({
      model: 'Doubao-Seedance-2.0',
      global_model_name: 'Doubao-Seedance-2.0',
      mapped_model: 'doubao-seedance-2-0-260128',
      observed_model: 'ep-20260717110243-mk4p4',
    })).toEqual({
      primary: 'Doubao-Seedance-2.0',
      requested: null,
      mapped: 'doubao-seedance-2-0-260128',
      observed: 'ep-20260717110243-mk4p4',
    })
  })

  it('shows a differing request identity without replacing the global identity', () => {
    expect(videoTaskModelIdentity({
      model: 'client-video-alias',
      global_model_name: 'Doubao-Seedance-2.0',
      mapped_model: 'doubao-seedance-2-0-260128',
      observed_model: null,
    })).toEqual({
      primary: 'Doubao-Seedance-2.0',
      requested: 'client-video-alias',
      mapped: 'doubao-seedance-2-0-260128',
      observed: null,
    })
  })

  it('falls back to the request model for records without identity metadata', () => {
    expect(videoTaskModelIdentity({
      model: 'legacy-video-model',
      global_model_name: null,
      mapped_model: ' ',
      observed_model: undefined,
    })).toEqual({
      primary: 'legacy-video-model',
      requested: null,
      mapped: null,
      observed: null,
    })
  })
})
