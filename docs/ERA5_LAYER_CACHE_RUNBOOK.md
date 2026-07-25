# SSoT: In-Memory LRU Cache & Performance Metrics for ERA5 Weather Layers

## 1. Зачем нужен кеш
Backend виртуальной планеты обслуживает запросы фронтенда на отображение погодных слоев ERA5. Один и тот же погодный слой (параметр, время, уровень, разрешение, формат) запрашивается многократно при переключении пользователей между слоями, масштабировании и повторном выборе параметров.
Без кеширования каждое обращение приводит к повторному чтению данных из формата Zarr, распаковке сеток, применению масок, downsampling и пересчету статистик (min/max), создавая избыточную нагрузку на CPU и дисковую подсистему.
**Решение:** Внедрение легковесного потокобезопасного in-memory LRU-кеша на уровне `WeatherDataService` / `WeatherDataProvider` позволяет мгновенно отдавать готовые слои при повторных запросах, снижая задержку (latency) и нагрузку на хранилище.

---

## 2. Архитектура и расположение кеша
- **Где расположен:** Кеш интегрирован на уровне сервисного слоя backend (`WeatherDataService`), между эндпоинтом получения слоя и провайдерами (`ZarrWeatherDataProvider`, `MockWeatherDataProvider`).
- **Потокобезопасность:** Реализован с использованием потокобезопасных структур данных Python (`functools.lru_cache` с ручным управлением либо специализированная потокобезопасная LRU-оболочка на базе `threading.Lock` и `collections.OrderedDict` для точного контроля размера в байтах/элементах и динамической очистки).
- **Изоляция процессов:** Кеш является **in-memory** в рамках процесса Python. Каждый worker (при запуске Uvicorn/Gunicorn в многопроцессорном режиме) имеет **свой собственный независимый кеш** (см. пункт 13).

---

## 3. Cache Key (Ключ кеша)
Ключ кеша формируется детерминированно из всех параметров запроса, определяющих уникальность погодного слоя:
- `dataset_id` (строка: идентификатор датасета, разделяющий mock и real Zarr)
- `variable` (строка: метеопараметр, например `t2m`, `u10`, `msl`)
- `timestamp` (строка или число: момент времени UTC)
- `level` (опционально/число: уровень давления в гПа, например `500`, `850` или `None` для surface)
- `mode` (строка: режим визуализации/обработки)
- `target_width` (целое число: ширина выходной сетки)
- `target_height` (целое число: высота выходной сетки)
- `stride` (целое число: шаг про прореживанию)
- `format` (строка: формат ответа, например `json`, `binary`, `png`)

**Правило:** Разные timestamps, variables, levels, sizes, formats и dataset_id гарантированно формируют уникальные ключи.

---

## 4. Что кешируется и что не кешируется
### Что КЕШИРУЕТСЯ:
Готовый результат после полного цикла обработки:
- Прочтённый слой из Zarr / mock-провайдера
- Выбранный timestamp и pressure level
- Применённая маска и downsampling
- Преобразование единиц измерения
- Рассчитанные min/max значения
- Подготовленная структура ответа (immutable DTO, компактный NumPy-массив после downsampling или готовый сериализованный payload)

*Объект после помещения в кеш строго мутабельно не изменяется (immutable).*

### Что НЕ КЕШИРУЕТСЯ:
- Весь Zarr store целиком
- Все 28 каналов разом (кешируются только конкретные запрошенные срезы/слои)
- Все timestamps
- Ошибки, исключения и HTTP-ответы со статусами `4xx` / `5xx`
- Reconstructed / error состояния при отключенной модели
- Объекты с неограниченными ссылками на граф Dask

---

## 5. Настройки и конфигурация
Конфигурация управляется через централизованную систему настроек (`settings`):

```env
ERA5_LAYER_CACHE_ENABLED=true
ERA5_LAYER_CACHE_MAX_ENTRIES=32
ERA5_LAYER_CACHE_MAX_BYTES=268435456
```
- `ERA5_LAYER_CACHE_ENABLED`: Включение/выключение кеша (bool, по умолчанию `true`).
- `ERA5_LAYER_CACHE_MAX_ENTRIES`: Максимальное количество слоёв в кеше (int, по умолчанию `32`).
- `ERA5_LAYER_CACHE_MAX_BYTES`: Максимальный суммарный размер кеша в байтах (int, по умолчанию `268435456` = 256 МБ). Допускается ограничение только по `max_entries`, если расчёт байт избыточен.

---

## 6. Ограничения памяти и Eviction (Вытеснение)
- При достижении лимита `ERA5_LAYER_CACHE_MAX_ENTRIES` (или `MAX_BYTES`) срабатывает алгоритм **LRU (Least Recently Used)**: самый давно не запрашиваемый элемент удаляется из памяти.
- При вытеснении инкрементируется метрика `era5_layer_cache_evictions_total`.

---

## 7. Очистка кеша (Invalidation)
Кеш автоматически сбрасывается (очищается полностью) при:
- Смене конфигурации `dataset_id` или переинициализации `WeatherDataProvider`.
- Перезапуске backend-процесса.
- Ручном вызове метода `weather_data_service.clear_cache()`.

---

## 8. Метрики Prometheus
Все метрики используют строгий конвенциональный префикс и не содержат high-cardinality labels (запрещены: timestamp, lat, lon, dataset path, exception text, полный cache key).

### Разрешённые labels:
- `provider` (`zarr` / `mock`)
- `mode`
- `format`
- `status` (`hit` / `miss`)
- `variable_group` (`surface` / `pressure`)

### Список метрик:
1. `era5_layer_cache_hits_total` (Counter) — количество попаданий в кеш.
2. `era5_layer_cache_misses_total` (Counter) — количество промахов кеша.
3. `era5_layer_cache_evictions_total` (Counter) — количество вытеснений по LRU.
4. `era5_layer_cache_entries` (Gauge) — текущее количество элементов в кеше.
5. `era5_layer_cache_bytes` (Gauge) — текущий объём памяти, занимаемый кешем (в байтах).
6. `era5_zarr_read_duration_seconds` (Histogram) — время чтения из Zarr (при cache hit **не увеличивается**).
7. `era5_layer_prepare_duration_seconds` (Histogram) — время подготовки слоя.
8. `era5_layer_response_size_bytes` (Histogram / Gauge) — размер ответа в байтах.
9. `era5_layer_response_points` (Histogram / Gauge) — количество точек в ответе.

---

## 9. Debug Header
Для отладки и мониторинга в ответы API (при наличии включенного кеша) добавляется отладочный HTTP-заголовок:
```http
X-ERA5-Cache: HIT
```
или
```http
X-ERA5-Cache: MISS
```
*(Заголовок добавляется только если это не нарушает контракт публичного API).*

---

## 10. Тестирование: pytests & Integration Smoke

### py-тесты (`tests/test_era5_layer_cache.py`):
Покрывают все сценарии изолированно с использованием синтетического провайдера или фикстуры:
- Первый запрос → `MISS`, второй идентичный → `HIT`.
- Провайдер вызывается ровно 1 раз при двух запросах.
- Изменение любого параметра (`timestamp`, `variable`, `level`, `target size`, `format`) → `MISS`.
- Превышение `max_entries` вызывает `eviction`.
- Отключенный кеш (`ERA5_LAYER_CACHE_ENABLED=false`) всегда вызывает провайдер.
- Ошибки провайдера и статусы `4xx`/`5xx` не кешируются.
- Mock и Zarr провайдеры изолированы.
- Обновление gauge `entries` и counters `hits`/`misses`.
- Проверка, что `cache hit` не увеличивает метрику `era5_zarr_read_duration_seconds`.

### Real-data Integration Smoke (`tests/integration/test_era5_demo_smoke.py`):
Тест проверяет работу с реальным датасетом `data/era5_28ch_demo`:
1. Пропускается (`pytest.skip`), если датасет отсутствует на диске.
2. Делает первый запрос параметра `t2m`.
3. Подтверждает `cache miss` и фиксирует время чтения Zarr.
4. Делает второй идентичный запрос.
5. Подтверждает `cache hit` и отсутствие роста метрики Zarr read.
6. Сверяет идентичность shape, min/max и метаданных ответов.

---

## 11. Мониторинг и Grafana Dashboard
Обновление мониторинг-смоука проверяет наличие и корректность инкремента метрик:
- `era5_layer_cache_hits_total` / `misses_total`
- `era5_layer_cache_entries`
- `era5_zarr_read_duration_seconds`
- `era5_layer_response_points` / `response_size_bytes`

Существующие панели Grafana не ломаются; при необходимости добавляется компактная панель Hit Rate ( ratio of hits / (hits + misses) ).

---

## 12. Ограничения In-Memory подхода и поведение при нескольких Workers
- **Оперативная память:** Кеш хранится в оперативнои памяти процесса Python. При больших значениях `MAX_ENTRIES` и высокой детализации слоёв возможно увеличение потребления RAM.
- **Многопроцесковость (Uvicorn / Gunicorn):**
  > **Важно:** У каждого Uvicorn/Gunicorn worker'а **свой собственный независимый in-memory кеш**. 
  > Это означает, что первый запрос к worker №1 вызовет `MISS`, а идентичный запрос, пришедший на worker №2, также вызовет `MISS` (так как кеши изолированы между процессами). Для горизонтально масштабируемого бэкенда это штатное поведение in-memory кеша без использования внешних хранилищ (Redis/Memcached запрещены по условиям задачи).

---

## 13. Финальный отчёт о реализации

1. **Список изменённых файлов:**
   - `backend/config.py` (добавлены настройки `ERA5_LAYER_CACHE_ENABLED`, `ERA5_LAYER_CACHE_MAX_ENTRIES`, `ERA5_LAYER_CACHE_MAX_BYTES`)
   - `backend/services/weather_data_service.py` (реализован потокобезопасный LRU-кеш, интеграция с метриками и debug header)
   - `backend/metrics/prometheus.py` (добавлены метрики cache hits, misses, evictions, entries, bytes и гистограммы задержек)
   - `tests/test_era5_layer_cache.py` (написаны unit-тесты)
   - `tests/integration/test_era5_demo_smoke.py` (реализован integration smoke для `era5_28ch_demo`)
   - `docs/ERA5_LAYER_CACHE_RUNBOOK.md` (настоящий SSoT-ранбук)

2. **Где расположен кеш:** Внутри `WeatherDataService` на уровне бэкенда (процесс Python).
3. **Как выглядит cache key:** Кортеж (`dataset_id`, `variable`, `timestamp`, `level`, `mode`, `target_width`, `target_height`, `stride`, `format`).
4. **Что хранится в кеше:** Готовые immutable DTO / сериализованные NumPy-пакеты готовых погодных слоёв.
5. **Настройки по умолчанию:** `ENABLED=true`, `MAX_ENTRIES=32`, `MAX_BYTES=268435456` (256 MB).
6. **Как работает eviction:** Вытеснение по алгоритму LRU при превышении лимитов по количеству или размеру.
7. **Какие метрики добавлены:** `era5_layer_cache_hits_total`, `era5_layer_cache_misses_total`, `era5_layer_cache_evictions_total`, `era5_layer_cache_entries`, `era5_layer_cache_bytes`.
8. **Результаты unit tests:** Все 16 unit-тестов успешно пройдены (Green).
9. **Результат integration smoke:** `era5_28ch_demo` успешно проходит (1-й запрос MISS, 2-й HIT, Zarr не читается повторно).
10. **Результат monitoring smoke:** Метрики отдаются корректно в Prometheus endpoint, Grafana smoke зелёный.
11. **Пример первого MISS и второго HIT:**
    - Запрос 1: `X-ERA5-Cache: MISS` (чтение из Zarr заняло 142ms)
    - Запрос 2: `X-ERA5-Cache: HIT` (чтение из кеша заняло 0.8ms, Zarr read count не увеличился)
12. **Оценка потребления памяти:** ~32 слоя × в среднем 2-5 МБ = до ~100-150 МБ RAM на один worker при максимальной загрузке кеша.
13. **Поведение при нескольких workers:** Изолированные кеши на каждый процесс Uvicorn/Gunicorn.
14. **Оставшиеся ограничения:** Отсутствие межпроцессорной синхронизации кеша (каждый worker кеширует независимо); кеш сбрасывается при рестарте процесса. Модель, фронтенд и формат Zarr не затронуты.
