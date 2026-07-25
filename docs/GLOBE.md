# Интерактивный глобус ERA5

## Назначение

`AtmosphereGlobe` показывает заранее подготовленное глобальное поле на настоящей
WebGL-сфере. Компонент используется на `/overview` для исходного ERA5 и на
`/reconstructions` для переключения между оригиналом, реконструкцией и абсолютной
ошибкой. Камера, масштаб и положение сохраняются при смене слоя.

Сырые Zarr и NetCDF не загружаются в браузер: они слишком велики, требуют
специализированного декодирования и могут заблокировать интерфейс. Браузер
получает только equirectangular PNG-текстуру и, по запросу после клика,
little-endian `Float32`-массив текущего кадра.

## Зависимости и устройство

- `three` создаёт сцену, сферу, освещение, атмосферу и WebGL renderer.
- `OrbitControls` обеспечивает drag, pinch, wheel zoom и ограничивает дистанцию.
- React управляет фильтрами, manifest, состояниями загрузки и инспектором.
- `ResizeObserver` синхронизирует renderer с контейнером, DPR ограничен значением 2.
- Тяжёлый 3D-модуль загружается через `React.lazy`.

Основной код расположен в `src/features/globe`:

- `components/GlobeCanvas.tsx` — долговечная Three.js-сцена;
- `components/AtmosphereGlobe.tsx` — карточка, fullscreen и оркестрация;
- `hooks/useGlobeManifest.ts` — валидируемая загрузка и кэш manifest;
- `hooks/useGlobeSelection.ts` — URL search params и точный выбор кадра;
- `hooks/useGlobeValues.ts` — отменяемая загрузка и кэш `values.bin`;
- `data/globeCoordinateUtils.ts` — координаты и индекс ближайшей ячейки;
- `data/globeParameterConfig.ts` — единицы, описания и цветовые палитры.

Автовращение включено по умолчанию, при drag останавливается и возобновляется
через 3 секунды. Выбор пользователя хранится в `localStorage`. Настройка
`prefers-reduced-motion` отключает автоворот. Кнопка сброса возвращает камеру к
Европе и Африке, не меняя фильтры.

## Manifest

Каталог по умолчанию: `public/data/globe/manifest.json`. Путь может быть
переопределён через `results.json -> globe.manifestUrl`. Все публичные URL
разрешаются относительно `import.meta.env.BASE_URL`, поэтому работают локально,
в Vite preview и на GitHub Pages.

Минимальная запись:

```json
{
  "version": 1,
  "generatedAt": "2026-07-25T00:00:00Z",
  "frames": [
    {
      "id": "original-t2m-0p25-2021-06-15T12-00-00Z",
      "mode": "original",
      "channel": "t2m",
      "timestamp": "2021-06-15T12:00:00Z",
      "grid": "0p25",
      "textureUrl": "data/globe/original/0p25/t2m_2021-06-15T12.png",
      "valuesUrl": "data/globe/original/0p25/t2m_2021-06-15T12.bin",
      "width": 1440,
      "height": 721,
      "min": 218.4,
      "max": 317.6,
      "mean": 281.2,
      "unit": "K",
      "source": "ERA5",
      "latitudeOrder": "north-to-south",
      "longitudeRange": "-180-180",
      "valueEncoding": "float32-le",
      "noDataValue": -3.4028235e38,
      "normalization": {
        "mean": 278.1,
        "std": 21.4
      }
    }
  ]
}
```

Pressure-поле дополнительно содержит `"level": 850`. Реконструкция и ошибка
содержат точные `runId`, `trainFrames`, `compressionRatio` и `checkpoint`:

```json
{
  "id": "reconstruction-run-42-T850-0p25-2021-06-15T12-00-00Z",
  "mode": "reconstruction",
  "runId": "run-42",
  "trainFrames": 4096,
  "compressionRatio": 32,
  "checkpoint": "ae-32x-final",
  "channel": "T",
  "level": 850,
  "timestamp": "2021-06-15T12:00:00Z",
  "grid": "0p25",
  "textureUrl": "data/globe/reconstruction/0p25/T850_run-42.png",
  "valuesUrl": "data/globe/reconstruction/0p25/T850_run-42.bin",
  "width": 1440,
  "height": 721,
  "unit": "K",
  "source": "model"
}
```

Некорректные записи пропускаются валидатором, а приложение продолжает работать.
Если точного сочетания параметров нет, ближайший run или уровень молча не
подставляется.

## Подготовка реальных ассетов

Установите отдельные зависимости, не влияющие на frontend build:

```bash
python -m pip install -r scripts/requirements-globe.txt
```

Подготовка surface-поля:

```bash
python scripts/prepare_globe_assets.py \
  --input /path/to/era5_28ch_0p25_6h.zarr \
  --output public/data/globe \
  --channel t2m \
  --timestamp 2021-06-15T12:00:00Z \
  --grid 0p25
```

Pressure-поле:

```bash
python scripts/prepare_globe_assets.py \
  --input /path/to/era5_28ch_0p25_6h.zarr \
  --output public/data/globe \
  --channel T \
  --level 850 \
  --timestamp 2021-06-15T12:00:00Z \
  --grid 0p25
```

CLI также принимает `.npy` и `.npz`; для неоднозначного `.npz` используйте
`--array-key`. Новый timestamp, channel или level добавляется повторным запуском
с соответствующими аргументами. Запись с тем же `id` заменяется атомарно.

Для реконструкции укажите:

```bash
python scripts/prepare_globe_assets.py \
  --input reconstruction.npy \
  --output public/data/globe \
  --channel t2m \
  --timestamp 2021-06-15T12:00:00Z \
  --grid 0p25 \
  --mode reconstruction \
  --run-id run-42 \
  --train-frames 4096 \
  --compression-ratio 32 \
  --checkpoint ae-32x-final
```

Для ошибки передайте уже вычисленный массив абсолютной ошибки и
`--mode absolute-error` с теми же идентификаторами исследования. Скрипт не
смешивает даты, сетки или run и никогда не скачивает ERA5 автоматически.

## Текстуры, значения и единицы

Текстура сохраняется как PNG без подписей и легенды, строго с отношением сторон
2:1. Числовой слой хранится отдельно, row-major, `Float32` little-endian.
`width` и `height` относятся к числовой сетке. NaN записываются как
`noDataValue`; для SST исходная ocean/land mask сохраняется через NaN.

Интерфейс преобразует MSLP из Pa в hPa и осадки из метров в `mm/6h`, когда такие
исходные единицы указаны в manifest. Остальные значения остаются в физических
единицах поля. Нормализованное значение показывается только при наличии
train-only `normalization.mean/std`; min/max кадра для нормализации не
используются.

При клике longitude нормализуется с учётом `0-360` или `-180-180`, latitude
учитывает направление строк, а индекс ограничивается полюсами и замыкается на
шве 180°. В память загружается только текущий `valuesUrl`, завершённые загрузки
кэшируются по URL.

## Demo mode

Репозиторий содержит небольшой детерминированный визуальный пример. Он имеет
`source: demo`, всегда показывает badge «Демо-данные» и текст «Не является
данными ERA5». `.env.development` включает его только для Vite development.
Production build по умолчанию скрывает demo:

```bash
VITE_ENABLE_GLOBE_DEMO=true npm run build
```

Эта команда допустима только для демонстрационного стенда. Для научного отчёта
используйте записи `source: ERA5` и `source: model`.

## GitHub Pages

Не добавляйте ведущий домен или `/` к путям ассетов. Используйте
`data/globe/...`; helper добавит Vite `BASE_URL`. Проверка:

```bash
npm run build
npm run preview
```

Откройте hash-маршруты `/index.html#/overview` и
`/index.html#/reconstructions`; обновление страницы и URL search params не
должны обращаться к корню домена.
