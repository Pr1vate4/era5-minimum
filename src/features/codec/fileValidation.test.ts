import { describe, expect, it } from 'vitest'
import { validateCodecFile } from './fileValidation'

describe('codec file validation', () => {
  it('accepts the supported ERA5 container formats', () => {
    for (const name of ['frame.npz', 'frame.npy', 'frame.nc', 'frame.zarr.zip']) {
      expect(validateCodecFile(new File(['era5'], name))).toBeNull()
    }
  })

  it('rejects unsupported and empty files', () => {
    expect(validateCodecFile(new File(['x'], 'frame.csv'))).toMatch(/формат/i)
    expect(validateCodecFile(new File([], 'frame.npz'))).toMatch(/пуст/i)
  })

  it('rejects browser uploads larger than two gibibytes', () => {
    const file = { name: 'huge.npz', size: 2 * 1024 ** 3 + 1 } as File
    expect(validateCodecFile(file)).toMatch(/2 ГиБ/i)
  })
})
