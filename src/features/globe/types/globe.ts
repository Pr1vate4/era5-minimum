export type GlobeMode = 'original' | 'reconstruction' | 'absolute-error'

export type GlobeGrid = '0p25' | '0p5'

export type SurfaceChannel =
  | 't2m'
  | 'mslp'
  | 'u10'
  | 'v10'
  | 'wind10'
  | 'tp6h'
  | 'sst'
  | 'tcwv'
  | 'tcc'

export type PressureVariable = 'T' | 'U' | 'V' | 'Z' | 'Q'

export type GlobeChannel = SurfaceChannel | PressureVariable

export type PressureLevel = 1000 | 925 | 850 | 700

export type GlobeSource = 'ERA5' | 'model' | 'demo'

export interface GlobeNormalization {
  mean: number
  std: number
}

export interface GlobeFrameAsset {
  id: string
  mode: GlobeMode
  runId?: string
  channel: GlobeChannel
  level?: PressureLevel
  timestamp: string
  grid: GlobeGrid
  textureUrl: string
  valuesUrl?: string
  width: number
  height: number
  min?: number
  max?: number
  mean?: number
  unit: string
  source: GlobeSource
  compressionRatio?: number
  trainFrames?: number
  checkpoint?: string
  latitudeOrder?: 'north-to-south' | 'south-to-north'
  longitudeRange?: '0-360' | '-180-180'
  valueEncoding?: 'float32-le'
  noDataValue?: number
  normalization?: GlobeNormalization
}

export interface GlobeManifest {
  version: number
  generatedAt?: string
  frames: GlobeFrameAsset[]
}

export interface GlobeResearchDefaults {
  runId?: string
  trainFrames?: number
  compressionRatio?: number
  checkpoint?: string
}

export interface GlobeCoordinates {
  latitude: number
  longitude: number
}

export interface GlobeGridPoint extends GlobeCoordinates {
  row: number
  column: number
  flatIndex: number
  value?: number
}

export type GlobeManifestState = {
  manifest: GlobeManifest | null
  loading: boolean
  error: string | null
  invalidFrameCount: number
  reload: () => void
}
