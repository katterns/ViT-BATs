# Классификация эхолокационных сигналов летучих мышей (NABat)

Технический README: откуда взять данные, как собрать локальный корпус и в каком порядке запускать скрипты обучения.

Подробное описание экспериментов — в [`FULL_DESCRIPTION.md`](reports/FULL_DESCRIPTION.md) и [`ARTICLE.md`](reports/ARTICLE.md).

## Требования

- Python 3.11+ (управление зависимостями через [uv](https://docs.astral.sh/uv/))
- ~50+ ГБ свободного места для полного NABat ML training release (zip-архивы по видам)

```bash
uv sync
```

Все команды ниже — из корня репозитория: `uv run python <script>.py`.

## 1. Исходный датасет

Официальный training release NABat Machine Learning V1.0:

- **ScienceBase:** [Training dataset for NABat Machine Learning V1.0](https://www.sciencebase.gov/catalog/item/627ed4b2d34e3bef0c9a2f30)
- **DOI:** [10.5066/P969TX8F](https://doi.org/10.5066/P969TX8F)
- **Статья:** Khalighifar et al., 2022 — [NABat ML](https://doi.org/10.1111/1365-2664.14280)

Скачайте zip-архивы по видам (`ANPA.zip`, `COTO.zip`, …) и файл кодов видов `NABat_Species_Codes.csv` (в репозитории лежит копия: [`NABat_Species_Codes.csv`](NABat_Species_Codes.csv)).

В релизе **35 классов** (34 вида + `NOISE`). Holdout test set в этот release **не входит**.

## 2. Локальный датасет `nabat_paper_31/`

Пути заданы в [`config.py`](config.py):

| Путь | Назначение |
|------|------------|
| `data/` | zip-архивы NABat ML release и/или распакованные WAV |
| `data/nabat_paper_31/trainval/` | train + val split (файлы и импульсы) |
| `data/nabat_paper_31/test/` | holdout test split |
| `data/nabat_paper_31/trainval/spec_cache/` | RGB-кэш для train/val |
| `data/nabat_paper_31/test/spec_cache/` | RGB-кэш для test |

Сборка датасета (80/10/10 по файлам, stratify по виду, seed=42):

```bash
# из zip-архивов в data/ (медленно: extract + gottbat)
uv run python scripts/build_nabat_paper_dataset.py --raw data --from-zip

# если WAV уже распакованы в data/<SPECIES>/
uv run python scripts/build_nabat_paper_dataset.py --scan-extracted --wav-dir data

# с pulse-balance (как в paper-style экспериментах)
uv run python scripts/build_nabat_paper_dataset.py --from-zip --balance-pulses
```

Итоговые CSV: `files_train.csv`, `files_val.csv`, `files_test.csv`, `pulses_train.csv`, `pulses_val.csv`, `pulses_test.csv`.

Подробности split и метрик — [`reports/test_eval_summary.md`](reports/test_eval_summary.md).

## 3. Предобработка и кэш (NABat v2)

Датасет: `data/nabat_paper_31/`.

Pipeline: детекция импульсов → NABat quality filter → RGB **3×100×100** ([`bat/data/nabat.py`](bat/data/nabat.py)).

Первый прогон медленный; дальше используются кэши в `data/nabat_paper_31/`:

```bash
# spec cache для train/val/test (nabat_paper_31)
uv run python scripts/precompute_specs.py

# быстрая проверка на 1000 примеров
uv run python scripts/precompute_specs.py --limit 1000

# waveform cache для sep v3 SSL (опционально)
uv run python scripts/precompute_waveforms.py
```

Кэши:
- `data/nabat_paper_31/trainval/spec_cache/`
- `data/nabat_paper_31/test/spec_cache/`
- `data/nabat_paper_31/trainval/waveform_cache/` (sep v3)

Визуализация предобработки и contrastive input для SSL:

```bash
uv run python scripts/visualize_preprocessed.py
uv run python scripts/generate_nabat_v2_report.py --skip-histogram
```

## 4. Обучение моделей

Чекпоинты и логи Lightning: `checkpoints/`, `checkpoints/lightning_logs/<run_name>/`.

### Supervised baselines

```bash
uv run python supervised_cnn_baseline.py
uv run python supervised_resnet_baseline.py
uv run python beats_ultrasound_baseline.py
```

### ViT + LFPE

```bash
uv run python vit_lfpe_ssl_pretrain.py

# Fine-tune с SSL-энкодером
uv run python vit_lfpe_bat_baseline.py

# Fine-tune без SSL (контроль)
uv run python vit_lfpe_bat_baseline.py --no-ssl
```

### CNN SSL ablations

```bash
# SSL pretrain + finetune (пример: mae v2)
uv run python ablations_cnn/pretrain.py --preset mae --ssl-version 2
uv run python ablations_cnn/finetune.py --preset mae --ssl-version 2

# test eval
uv run python scripts/eval_test_summary.py --preset mae --ssl-version 2
```

### Отчёты и оценка

```bash
# гистограмма macro-F1 + contrastive preview (без повторного обучения)
uv run python scripts/generate_nabat_v2_report.py
```

Сводка метрик: [`reports/test_eval_summary.md`](reports/test_eval_summary.md) (test), [`reports/eval_summary.md`](reports/eval_summary.md) (legacy val).

## 5. Конфигурация

Основные параметры — [`config.py`](config.py):

| Параметр | Значение |
|----------|----------|
| `RANDOM_SEED` | 42 |
| `NABAT_PAPER_DIR` | `data/nabat_paper_31` |
| `SPEC_CACHE_VERSION` | `nabat_v2_f16_100` |

## 6. Порядок с нуля

1. `uv sync`
2. Скачать zip с [ScienceBase](https://www.sciencebase.gov/catalog/item/627ed4b2d34e3bef0c9a2f30) → `data/`
3. `uv run python scripts/build_nabat_paper_dataset.py --from-zip --balance-pulses`
4. `uv run python scripts/precompute_specs.py`
5. Обучить модели (§4)
6. `uv run python scripts/eval_test_summary.py`

## Ссылки

- [NABat ML training dataset (USGS)](https://www.sciencebase.gov/catalog/item/627ed4b2d34e3bef0c9a2f30)
- [Khalighifar et al., 2022](https://doi.org/10.1111/1365-2664.14280)
- [A Guide to Processing Bat Acoustic Data for NABat](https://doi.org/10.3133/ofr20181068)
