export type NumericValue = number | string

export type Meta = {
  team?: string
  run_id?: string
  grid?: string
  checkpoint?: string
  generated_at?: string
  training_samples?: NumericValue
  gpu_name?: string
  training_duration?: string
  environment?: string
  external_pretraining?: boolean
  batch_size?: NumericValue
  processed_examples?: NumericValue
}

export type Compression = {
  target_ratio?: NumericValue
  actual_ratio?: NumericValue
  bitstream_bytes?: NumericValue
  roundtrip_exact?: boolean
}

export type Scores = {
  surface_nrmse?: NumericValue
  pressure_nrmse?: NumericValue
  overall_nrmse?: NumericValue
  delta_vs_reference_pct?: NumericValue
  delta_ci95?: [NumericValue, NumericValue]
  psnr_delta_db?: NumericValue
  spectral_error_pct?: NumericValue
}

export type Criterion = {
  name: string
  target?: string
  value?: NumericValue | boolean
  unit?: string
  pass?: boolean
  compression_ratio?: NumericValue
}

export type PerChannel = {
  code: string
  group?: 'surface' | 'pressure'
  name?: string
  unit?: string
  rmse?: NumericValue
  nrmse?: NumericValue
  psnr?: NumericValue
  level_hpa?: NumericValue
  delta_vs_reference_pct?: NumericValue
  pass?: boolean
}

export type DataEfficiencyPoint = {
  n_samples?: NumericValue
  overall_nrmse?: NumericValue
  surface_nrmse?: NumericValue
  pressure_nrmse?: NumericValue
  reference_nrmse?: NumericValue
  ci_low?: NumericValue
  ci_high?: NumericValue
  pass?: boolean
}

export type RateDistortionPoint = {
  compression_ratio?: NumericValue
  overall_nrmse?: NumericValue
  surface_nrmse?: NumericValue
  pressure_nrmse?: NumericValue
  psnr?: NumericValue
  reference_nrmse?: NumericValue
}

export type SpectralPoint = {
  wavenumber?: NumericValue
  reference_energy?: NumericValue
  model_energy?: NumericValue
  channel?: string
  grid?: string
  compression_ratio?: NumericValue
}

export type ProbeForecast = {
  n_pairs?: NumericValue
  steps?: NumericValue
  params_millions?: NumericValue
  latent_nrmse?: NumericValue
  persistence_nrmse?: NumericValue
  reference_probe_nrmse?: NumericValue
  improvement_vs_persistence_pct?: NumericValue
  improvement_vs_reference_probe_pct?: NumericValue
  pass?: boolean
}

export type Resources = {
  gpu_count?: NumericValue
  gpu_hours?: NumericValue
  peak_vram_gb?: NumericValue
  params_millions?: NumericValue
  optimizer_steps?: NumericValue
  probe_params_millions?: NumericValue
  probe_steps?: NumericValue
}

export type ReconstructionStats = {
  min?: NumericValue
  max?: NumericValue
  mean?: NumericValue
}

export type Reconstruction = {
  timestamp: string
  channel: string
  original_image?: string
  reconstructed_image?: string
  diff_image?: string
  relative_error_image?: string
  unit?: string
  level_hpa?: NumericValue
  grid?: string
  compression_ratio?: NumericValue
  original_stats?: ReconstructionStats
  reconstructed_stats?: ReconstructionStats
  error_stats?: ReconstructionStats
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
