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

## 2. Локальный корпус `cleaned/`

Пути заданы в [`config.py`](config.py):

| Путь | Назначение |
|------|------------|
| `data/nabat_raw/` | скачанные zip с ScienceBase (не в git) |
| [`data/audio_metadata_cleaned.csv`](data/audio_metadata_cleaned.csv) | **индекс после фильтрации** (в git, 17 122 записи, 26 классов) |
| `cleaned/` | распакованные WAV (~17k файлов, не в git) |
| `cleaned/audio_metadata_cleaned.csv` | копия индекса рядом с WAV |
| `cleaned_subset_200/` | подвыборка для supervised (≤200 на класс) |

Папки `cleaned*` в `.gitignore` — WAV не коммитятся; в git хранится только отфильтрованный манифест в `data/`.

### 2.1. Что убрано из исходного release

**Малочисленные классы** (USGS data release — малый N):

| Код | Вид |
|-----|-----|
| CORA | *Corynorhinus rafinesquii* |
| EUFL | *Eumops floridanus* |
| LAXA | *Lasiurus xanthinus* |
| NYFE | *Nyctinomops femorosaccus* |

**Дополнительно исключены** при сборке `cleaned/`:

`EUPE`, `IDPH`, `MYAU`, `MYVE`, `NYHU`

**Осталось 26 классов:**

`ANPA`, `COTO`, `EPFU`, `EUMA`, `LABL`, `LABO`, `LACI`, `LAIN`, `LANO`, `LASE`, `MYCA`, `MYCI`, `MYEV`, `MYGR`, `MYLE`, `MYLU`, `MYSE`, `MYSO`, `MYTH`, `MYVO`, `MYYU`, `NOISE`, `NYMA`, `PAHE`, `PESU`, `TABR`

### 2.2. Фильтр по длительности

**0,52–8,09 с** (min ≈ 0,523 с, max ≈ 8,090 с в манифесте).

### 2.3. Сборка `cleaned/` из zip

1. Скачайте zip-архивы в `data/nabat_raw/` (по одному на вид: `ANPA.zip`, `COTO.zip`, …).

2. Запустите скрипт сборки:

```bash
# полная сборка: сканирование zip → фильтрация → CSV + распаковка WAV
uv run python scripts/build_cleaned.py
```

Опции:

```bash
# только пересобрать манифест (без WAV)
uv run python scripts/build_cleaned.py --manifest-only

# распаковать по готовому манифесту из data/ (если zip уже скачаны)
uv run python scripts/build_cleaned.py --from-manifest

# свои пути
uv run python scripts/build_cleaned.py \
  --raw ./data/nabat_raw \
  --out ./cleaned \
  --manifest ./data/audio_metadata_cleaned.csv
```

Скрипт [`scripts/build_cleaned.py`](scripts/build_cleaned.py):

- обходит `*.zip` в `--raw`;
- пропускает 9 исключённых классов;
- читает длительность каждого WAV через `soundfile`;
- оставляет записи с длительностью **0,52–8,09 с**;
- пишет `data/audio_metadata_cleaned.csv` и копию в `cleaned/`;
- распаковывает WAV в `cleaned/<SPECIES>/`.

Если `cleaned/` уже собран локально, можно использовать готовый манифест из репозитория:

```bash
cp data/audio_metadata_cleaned.csv cleaned/audio_metadata_cleaned.csv
uv run python scripts/build_cleaned.py --from-manifest
```

### 2.4. Формат `audio_metadata_cleaned.csv`

```csv
species,filename,duration,sample_rate,archive
ANPA,ANPA-103486.wav,7.004,256000,ANPA.zip
```

| Колонка | Описание |
|---------|----------|
| `species` | четырёхбуквенный код |
| `filename` | имя WAV |
| `duration` | длительность, с |
| `sample_rate` | Hz |
| `archive` | исходный zip из ScienceBase |

## 3. Подвыборка `cleaned_subset_200/`

Для сравнимых supervised-экспериментов (5116 файлов, ≤200 на класс, `random_state=42`):

```bash
uv run python scripts/build_cleaned_subset_200.py
```

Опции:

```bash
uv run python scripts/build_cleaned_subset_200.py \
  --cleaned ./cleaned \
  --out ./cleaned_subset_200 \
  --cap 200
```

Скрипт читает `cleaned/audio_metadata_cleaned.csv`, сэмплирует до `--cap` записей на вид, копирует WAV из `cleaned/<species>/` или распаковывает из zip.

Итог: **5116** файлов, **26** классов, train/val **85/15** (стратификация, seed `42`) — см. [`bat/data/splits.py`](bat/data/splits.py).

## 4. Предобработка и кэш (NABat v2)

Текущий pipeline: детекция импульсов → NABat quality filter → RGB **3×100×100** ([`bat/data/nabat.py`](bat/data/nabat.py)).

Первый прогон медленный; дальше используется кэш в `checkpoints/spec_cache/`:

```bash
# subset для CNN / ResNet / ViT fine-tune (рекомендуется)
uv run python scripts/precompute_specs.py --subset

# полный cleaned (опционально, для SSL на full corpus)
uv run python scripts/precompute_specs.py --full

# быстрая проверка на 1000 примеров
uv run python scripts/precompute_specs.py --subset --limit 1000
```

Визуализация предобработки и contrastive input для SSL:

```bash
uv run python scripts/visualize_preprocessed.py
uv run python scripts/generate_nabat_v2_report.py --skip-histogram
```

## 5. Обучение моделей

Чекпоинты и логи Lightning: `checkpoints/`, `checkpoints/lightning_logs/<run_name>/`.

### Supervised baselines (subset_200)

```bash
uv run python supervised_cnn_baseline.py
uv run python supervised_resnet_baseline.py
uv run python beats_ultrasound_baseline.py
```

### ViT + LFPE

```bash
# SSL pretrain (по умолчанию на subset_200; --full для cleaned/)
uv run python vit_lfpe_ssl_pretrain.py

# Fine-tune с SSL-энкодером
uv run python vit_lfpe_bat_baseline.py

# Fine-tune без SSL (контроль)
uv run python vit_lfpe_bat_baseline.py --no-ssl
```

### Отчёты и оценка

```bash
# гистограмма macro-F1 + contrastive preview (без повторного обучения)
uv run python scripts/generate_nabat_v2_report.py

# OOD inference: файлы из full cleaned, которых нет в subset_200
uv run python scripts/eval_cnn_ood.py
```

Сводка метрик NABat v2: [`checkpoints/eval_summary.md`](checkpoints/eval_summary.md).

## 6. Конфигурация

Основные параметры — [`config.py`](config.py):

| Параметр | Значение |
|----------|----------|
| `RANDOM_SEED` | 42 |
| `VAL_TEST_SIZE` | 0.15 |
| `USE_NABAT_PREPROCESSING` | True |
| `NABAT_QUALITY_FILTER` | True |
| `SPEC_CACHE_VERSION` | `nabat_v2_quality_100` |

## 7. Порядок с нуля

1. `uv sync`
2. Скачать zip с [ScienceBase](https://www.sciencebase.gov/catalog/item/627ed4b2d34e3bef0c9a2f30) → `data/nabat_raw/`
3. `uv run python scripts/build_cleaned.py` — манифест + `cleaned/`
4. `uv run python scripts/build_cleaned_subset_200.py` — `cleaned_subset_200/`
5. `uv run python scripts/precompute_specs.py --subset`
6. Обучить модели (§5)
7. `uv run python scripts/generate_nabat_v2_report.py`

## Ссылки

- [NABat ML training dataset (USGS)](https://www.sciencebase.gov/catalog/item/627ed4b2d34e3bef0c9a2f30)
- [Khalighifar et al., 2022](https://doi.org/10.1111/1365-2664.14280)
- [A Guide to Processing Bat Acoustic Data for NABat](https://doi.org/10.3133/ofr20181068)
