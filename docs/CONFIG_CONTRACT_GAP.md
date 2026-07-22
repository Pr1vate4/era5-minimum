# Расхождение config и data contract

`configs/research_template.yaml` сохранён без изменений, потому что текущий `src/era5_minimum/experiments.py` поддерживает только `data.source=synthetic`. Config содержит legacy `tp6h`, `mslp`, single-file path `data/prepared_era5.nc` и `source=netcdf`; его безопасное переименование без реализации real-data experiment runner создало бы ложное впечатление готовности обучения на реальных данных.

`configs/mvp.yaml` также содержит `mslp` и `tp6h`, но это historical synthetic MVP vocabulary, а не schema реальных raw files. Он остаётся runnable только для synthetic generator и не должен использоваться как real-data config.

Это не второй data contract. Актуальная schema определена только в [DATA_CONTRACT.md](DATA_CONTRACT.md).

Будущий PR должен одновременно реализовать поддержку real loader → temporal split → train-only normalization в experiment runner, определить config schema и заменить legacy channel names после integration tests. До этого `research_template.yaml` — historical planned template, не runnable config.
