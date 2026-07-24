export type Meta = {
  team: string
  run_id: string
  grid: string
  checkpoint: string
  generated_at: string
}

export type Compression = {
  target_ratio: number
  actual_ratio: number
  bitstream_bytes: number
  roundtrip_exact: boolean
}

export type Scores = {
  surface_nrmse: number
  pressure_nrmse: number
  overall_nrmse: number
  delta_vs_reference_pct: number
  delta_ci95: [number, number]
  psnr_delta_db: number
  spectral_error_pct: number
}

export type Criterion = {
  name: string
  target: string
  value: number | boolean
  unit: string
  pass: boolean
}

export type PerChannel = {
  code: string
  group: 'surface' | 'pressure'
  name: string
  unit: string
  rmse: number
  nrmse: number
  psnr: number
}

export type DataEfficiencyPoint = {
  n_samples: number
  overall_nrmse: number
}

export type RateDistortionPoint = {
  compression_ratio: number
  overall_nrmse: number
}

export type SpectralPoint = {
  wavenumber: number
  reference_energy: number
  model_energy: number
}

export type ProbeForecast = {
  n_pairs: number
  steps: number
  improvement_vs_persistence_pct: number
  improvement_vs_reference_probe_pct: number
}

export type Resources = {
  gpu_hours: number
  peak_vram_gb: number
  params_millions: number
  optimizer_steps: number
}

export type Reconstruction = {
  timestamp: string
  channel: string
  original_image: string
  reconstructed_image: string
  diff_image: string
}

export type DashboardResults = {
  meta: Meta
  compression: Compression
  scores: Scores
  criteria: Criterion[]
  per_channel: PerChannel[]
  data_efficiency: DataEfficiencyPoint[]
  rate_distortion: RateDistortionPoint[]
  spectral: SpectralPoint[]
  probe_forecast: ProbeForecast
  resources: Resources
  reconstructions: Reconstruction[]
}
