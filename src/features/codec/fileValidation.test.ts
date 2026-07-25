import { describe, expect, it } from 'vitest'
import { validateCodecFile } from './fileValidation'

describe('codec file validation', () => {
  it('accepts a canonical NPZ container', () => {
    expect(validateCodecFile(new File(['era5'], 'frame.npz'))).toBeNull()
  })

  it('rejects unsupported and empty files', () => {
    expect(validateCodecFile(new File(['x'], 'frame.npy'))).toMatch(/формат/i)
    expect(validateCodecFile(new File([], 'frame.npz'))).toMatch(/пуст/i)
  })

  it('rejects browser uploads larger than 64 mebibytes', () => {
    const file = { name: 'huge.npz', size: 64 * 1024 ** 2 + 1 } as File
    expect(validateCodecFile(file)).toMatch(/64 МиБ/i)
  })
})
