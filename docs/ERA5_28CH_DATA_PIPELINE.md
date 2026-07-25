# Конвейер 28-канальных данных ERA5

Канонический источник — анонимный ERA5 Zarr WeatherBench2, объявленный в
`era5_minimum.data.weatherbench2.WB2_URL`. Он содержит восемь приземных полей
`t2m, mslp, u10, v10, tp6h, sst, tcwv, tcc` и T/U/V/Z/Q на уровнях
1000, 925, 850 и 700 hPa. Полный порядок модели задан `CHANNEL_NAMES` в
`src/era5_minimum/data/channel_spec.py`; адаптер создаёт ленивый тензор
`[time, channel, latitude, longitude]`.

Установите зависимости данных из `pyproject.toml`, затем запросите только
метаданные удалённого источника:

```bash
python scripts/data/prepare_era5_28ch.py inspect
```

Нативная директория 0.25° содержит `train.zarr`, `validation.zarr`,
`test.zarr`, `static.zarr`, `manifest.json`, `manifest.json.sha256` и
`_SUCCESS` только после завершения всех трёх именованных split. Динамические
поля имеют `float32` и chunks `time=4, latitude=180, longitude=180, level=4`.
В физическом хранилище SST остаётся `NaN` над сушей. `ocean_mask`, если она
доступна, равна `land_sea_mask <= 0.5`; некорректный SST можно заменить нулём
только в нормализованном ML batch и только вместе с отдельной маской.

Безопасный удалённый smoke-диапазон — не полный split:

```bash
python scripts/data/prepare_era5_28ch.py download --split validation \
  --start 2020-01-01 --end 2020-01-01 --output-dir data/era5_28ch_smoke
python scripts/data/prepare_era5_28ch.py validate --dataset-dir data/era5_28ch_smoke
python scripts/data/prepare_era5_28ch.py layer --dataset-dir data/era5_28ch_smoke \
  --split validation --variable T1000 --pressure-level-hpa 1000 --timestamp 2020-01-01T00:00:00
```

Полные split запускаются только явными командами и могут быть очень большими:

```bash
python scripts/data/prepare_era5_28ch.py download --split validation
python scripts/data/prepare_era5_28ch.py download --split test
python scripts/data/prepare_era5_28ch.py download --split train
python scripts/data/prepare_era5_28ch.py validate
python scripts/data/prepare_era5_28ch.py statistics
python scripts/data/prepare_era5_28ch.py remap --source data/era5_28ch_0p25_6h --output data/era5_28ch_0p5_6h
```

`statistics` делает ленивые Dask-редукции среднего, стандартного отклонения и
числа значений только по train; validation/test не читаются. `remap` должен
падать безопасно, пока не реализован и не проверен обязательный консервативный
ремаппинг первого порядка: нельзя заменять его nearest, subsampling, coarsen
или bilinear-приближением.

Полный нативный timestamp на 0.25° занимает примерно
`28 × 721 × 1440 × 4` байт, то есть 116 МиБ до Zarr-сжатия. Шесть лет по четыре
кадра в сутки дают около 1 ТиБ raw-данных без metadata и static fields. Перед
загрузкой нужны несколько ТиБ рабочего места и явное подтверждение команды.

`get_layer` читает только один timestamp/variable/level и возвращает координаты,
значения, finite-mask, экстремумы и число valid точек для будущего backend. Его
нельзя использовать для возврата целых split или сериализации `NaN` напрямую в
JSON.
