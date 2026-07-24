# Спецификация Patch-based PCA baseline для сжатия метеорологических данных
**Проект:** ERA5-Minimum  
**Версия:** 0.1.0  

## 1. Назначение
Настоящий документ определяет архитектуру, математический аппарат и формат артефактов для baseline-модели сжатия метеорологических полей ERA5 на основе Patch-based PCA.
PCA не является нейросетью. Он используется как референсный baseline для последующего сравнения со свёрточным автокодировщиком (ConvAE):
* Patch PCA 32× ↔ ConvAE 32×
* Patch PCA 64× ↔ ConvAE 64×

Документ описывает:
* Пайплайн обработки данных;
* Логику разбиения на патчи и обработки padding;
* Алгоритм инкрементального обучения (IncrementalPCA);
* Формат сохраняемых артефактов;
* CLI-интерфейсы для обучения и оценки;
* Ограничения метода.

## 2. Область применения
Спецификация применяется для:
* Обучения baseline-моделей PCA;
* Оценки качества восстановления полей;
* Расчёта коэффициентов сжатия (payload и end-to-end);
* Формирования JSON-артефактов для backend API.

**Запрещено в рамках данной спецификации:**
* Реализация нейросетей (ConvAE, PyTorch);
* Использование full-map PCA (глобальный PCA по всей карте);
* Использование перекрывающихся патчей (overlapping patches);
* Агрегация осадков (`tp1h` → `tp6h`);
* Создание специализированных моделей для осадков.

## 3. Общий пайплайн обработки
Данные проходят следующий конвейер:
1. Метеорологическая карта `[time, channel, lat, lon]`
2. Temporal split (Train / Validation / Test)
3. Потоковая нормализация (только по Train)
4. Разбиение на неперекрывающиеся патчи
5. Преобразование патчей в векторы
6. PCA transform (сжатие)
7. PCA inverse transform (восстановление)
8. Сборка карты из патчей
9. Удаление padding
10. Денормализация
11. Расчёт метрик (с учётом SST mask и padding mask)

## 4. Работа с патчами (Patching)
### 4.1. Форматы данных
* **Вход:** `[batch, channel, height, width]`
* **Выход (после разбиения):** `[number_of_patches, patch_features]`
  где `patch_features = channel_count × patch_height × patch_width`

### 4.2. Правила разбиения и Padding
Текущая сетка ERA5 имеет размер `361 × 720`. Высота (361) не кратна стандартному патчу 8×8.
**Строгие правила:**
1. Запрещено молча отбрасывать строки или столбцы.
2. Исходная форма должна быть сохранена.
3. Добавляется **zero padding** справа и снизу до размера, кратного `patch_size` (выполняется *после* нормализации).
4. После восстановления padding удаляется.
5. **Требование:** `reconstructed.shape == original.shape`.
6. **Метрики:** Padding не должен участвовать в расчёте MSE, RMSE, MAE. Необходимо использовать маску исходной области.

## 5. Модель и обучение (IncrementalPCA)
Для предотвращения переполнения RAM используется `sklearn.decomposition.IncrementalPCA`.

### 5.1. Batching и partial_fit
Патчи передаются в `partial_fit` порциями (`incremental_batch_size`).
**Обработка последнего батча:**
* Число samples в `partial_fit` должно быть `>= n_components`.
* Последний батч может оказаться меньше `n_components`.
* **Запрещено** передавать его напрямую или отбрасывать.
* **Решение:** Использовать буфер. Остаток накапливается и объединяется с предыдущей порцией либо обрабатывается безопасным способом.

### 5.2. Temporal split и защита от Data Leakage
Разделение выполняется **строго по времени** (ранние → train, следующие → val, последние → test).
* Перемешивание timestamps до split **запрещено**.
* PCA подгоняется (`fit`) **только на train**.
* Train statistics (mean, std) используются для нормализации val и test.

## 6. Нормализация и обработка пропусков (NaN / SST)
### 6.1. Потоковая нормализация
Для каждого канала на train-части вычисляются `mean`, `std`, `valid_count`.
* Используется численно устойчивый streaming-алгоритм (например, Welford).
* Если `std == 0`, используется безопасное значение (фиксируется в metadata).

### 6.2. SST mask и NaN
Канал `sst` содержит NaN над сушей. PCA не поддерживает NaN.
1. Невалидный SST исключается из расчёта нормализации.
2. Перед PCA невалидные значения заменяются на `0` (техническая замена, не физическое восстановление).
3. SST mask сохраняется.
4. При расчёте метрик значения над сушей **игнорируются**.
5. Аналогичная логика применяется к общей validity mask для других каналов.

## 7. Конфигурация и коэффициенты сжатия
### 7.1. Параметры конфигурации
| Параметр | Описание |
| :--- | :--- |
| `patch_height`, `patch_width` | Размер патча (по умолчанию 8×8) |
| `target_compression_ratio` | Целевой коэффициент сжатия (32 или 64) |
| `n_components` | Явное число компонент (приоритетнее target) |
| `incremental_batch_size` | Размер батча для `partial_fit` |
| `train_fraction`, `validation_fraction`, `test_fraction` | Доли временного разделения |

Если задан только `target_compression_ratio`:
`n_components = floor(patch_feature_count / target_compression_ratio)` (минимум 1).

### 7.2. Коэффициенты сжатия
Система обязана рассчитывать и сохранять два независимых показателя:

**1. Payload compression ratio**
Отношение исходного патча к его PCA-коэффициентам (без учёта веса модели).
`payload_ratio = patch_feature_count / n_components`
*(Пример: 512 / 16 = 32×)*

**2. End-to-end compression ratio**
Учитывает все данные, необходимые для декодирования.
`e2e_ratio = original_bytes / (coefficients_bytes + pca_model_bytes + normalization_bytes + metadata_bytes)`
*Запрещено заявлять сжатие 32×/64× только на основании имени конфига.*

## 8. Формат артефактов
Результаты работы сохраняются в `output_dir`.

### 8.1. pca_model.npz
| Поле | Описание |
| :--- | :--- |
| `components`, `mean`, `singular_values` | Параметры PCA |
| `explained_variance`, `explained_variance_ratio` | Дисперсия |
| `n_components`, `n_samples_seen` | Метаданные обучения |
| `patch_height`, `patch_width`, `channel_count` | Параметры архитектуры |

### 8.2. normalization.json
```json
{
  "channel_names": ["u10", "v10", ...],
  "mean": [],
  "std": [],
  "valid_count": [],
  "source_split": "train"
}
```

### 8.3. Reconstruction Samples
Сохраняется ограниченное число примеров (`reconstruction_samples.npz` + JSON для API).
JSON должен соответствовать контракту `ARTIFACT_API_CONTRACT.md` и содержать:
`experiment_id`, `model="patch_pca"`, `metrics`, `channel`, `timestamp`, `is_demo` (или признак реального эксперимента).

## 9. CLI интерфейсы
### 9.1. fit_pca_baseline.py
**Назначение:** Обучение модели и сохранение артефактов.
**Пример:**
```bash
python scripts/fit_pca_baseline.py \
  --config configs/patch_pca_32x.yaml \
  --input outputs/real_era5/week_2024_01_02_2024_01_08 \
  --output-dir outputs/models/patch_pca_32x \
  --smoke-test
```
**Режим `--smoke-test`:** Работает на CPU, генерирует synthetic dataset с NaN, выполняет полный цикл fit/encode/decode/metrics без интернета и реальных данных.

### 9.2. evaluate_pca_baseline.py
**Назначение:** Загрузка обученной модели, оценка на val/test, формирование метрик и JSON-артефактов для backend.
**Пример:**
```bash
python scripts/evaluate_pca_baseline.py \
  --model outputs/models/patch_pca_32x \
  --input outputs/real_era5/week_2024_01_02_2024_01_08 \
  --output-dir outputs/evaluation/patch_pca_32x
```

## 10. Тестирование
Тесты (`pytest`) используют `tmp_path` и synthetic NPZ. Использование реальных данных из `data/` и `outputs/` **запрещено**.

**Обязательные сценарии (30 пунктов):**
1. **Патчи:** Разбиение, сборка, round-trip без PCA, padding (361×720, нечётные размеры, кратные размеры), разное число каналов.
2. **PCA:** Расчёт `n_components`, payload ratio, IncrementalPCA batching, обработка последнего маленького батча, encode/decode.
3. **Данные:** Temporal split без перемешивания, train-only normalization, защита от data leakage.
4. **Маски:** SST mask, NaN handling, маскирование padding в метриках.
5. **Артефакты:** Сохранение/загрузка модели, несовпадающий channel order/coordinates, дублирующиеся timestamps.
6. **Smoke:** Smoke fit, smoke evaluation, конечность всех метрик.

## 11. Ограничения метода и известные проблемы
1. **Осадки (`tp1h`):** PCA плохо восстанавливает разреженные поля с локальными экстремумами. В рамках baseline не применяются log1p, классификация или специализированные loss-функции. Ограничение фиксируется в метаданных.
2. **Размер модели:** End-to-end сжатие для маленьких выборок будет значительно ниже payload-сжатия из-за веса PCA-модели.
3. **Статус:** PCA является исключительно baseline для подготовки pipeline. Финальная конкурсная модель — нейросеть (ConvAE).
4. **Совместимость:** Официальный датасет может потребовать написания отдельного data adapter. Текущий loader работает с локальными NPZ.
