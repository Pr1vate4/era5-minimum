# Roadmap до победы

Актуальная ERA5 schema и ограничения raw input определены в [DATA_CONTRACT.md](DATA_CONTRACT.md).

## Этап 0 — постановка

Зафиксировать:
- точный формат данных организаторов;
- доступную GPU;
- критерий качества;
- определение коэффициента сжатия;
- разрешённые поля и сетку;
- необходимость реального bitstream.

## Этап 1 — надёжный baseline

- подготовленный ERA5 loader;
- train-only normalization;
- хронологический split;
- ConvAE 32×;
- метрики по каждой переменной;
- визуализация original/reconstruction/error.

Выход: один полностью воспроизводимый эксперимент на реальных данных.

## Этап 2 — исследование sample efficiency

Train sizes задаются числом временных срезов. Для каждого размера:
- одинаковая архитектура;
- одинаковый validation/test;
- несколько seeds при наличии времени;
- фиксированный бюджет эпох или шагов;
- confidence interval либо разброс результатов.

Сравниваем стратегии отбора:
1. contiguous — последовательный интервал;
2. random — случайные часы;
3. seasonal — сбалансировано по сезонам/месяцам;
4. diversity — отбор разных погодных режимов.

## Этап 3 — усиление модели

По приоритету:
1. residual blocks;
2. channel-weighted loss;
3. gradient/extreme-aware loss;
4. VAE latent regularization;
5. lightweight attention at bottleneck;
6. quantization-aware training.

Нельзя переходить дальше, пока baseline не воспроизводится.

## Этап 4 — реальное сжатие

- квантизация latent;
- сериализация;
- zstd baseline;
- при возможности entropy model/arithmetic coding;
- отдельные метрики tensor ratio и file ratio.

## Этап 5 — продукт и защита

Интерактивное демо:
- выбор временного среза и переменной;
- original / reconstructed / absolute error;
- train size slider;
- compression ratio;
- per-variable metric table;
- график качества от sample size;
- автоматическая рекомендация минимального N.

## Победная формула

Рабочая модель + честный эксперимент + сильный научный вывод + понятная визуализация + воспроизводимость.
