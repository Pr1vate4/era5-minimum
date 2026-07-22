# Roadmap

Актуальная schema и ограничения raw input определены в [DATA_CONTRACT.md](DATA_CONTRACT.md); задачи с owner и acceptance criteria — в [TASKS.md](TASKS.md).

| Этап | Status | Результат |
| --- | --- | --- |
| Case selection | completed | Цель исследования, честное различие tensor/file ratio и initial synthetic MVP. |
| ERA5 schema research | completed | Подтверждены raw pair, coordinates, units, SST mask и precipitation semantics. |
| Real loader | completed | Inspector, strict pair loader, `tp → tp1h` в loader и 0.5° subsampling. |
| Team Bootstrap | completed | Python 3.12 CI, workflow, templates, onboarding, security policy и offline downloader checks. |
| Safe downloader | completed | Один CDS CLI, dry-run, staging, ZIP safety, provenance и SHA-256. |
| Extended real dataset | next | DATA-002: verified sequential hourly ERA5 range в shared storage. |
| Split and normalization | planned | DATA-003 после DATA-002, без temporal leakage и с train-only statistics. |
| Baselines | planned | PCA и ConvAE 32× на достаточных real data. |
| Sample-efficiency research | planned | Fixed validation/test, controlled train sizes и documented selection strategies. |
| Product/demo | planned | Visualization contract, mock dashboard, затем API/frontend/monitoring по отдельным задачам. |

Нельзя переходить к честному real-data baseline или research, пока DATA-002 и DATA-003 не приняты. Текущие четыре timestamps не являются training dataset.
