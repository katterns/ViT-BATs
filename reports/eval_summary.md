# Сводка результатов (NABat v2, val inference)

Дата: 2026-06-30 (CNN/ResNet), 2026-07-09 (ViT)  
Обучение не запускалось — только inference на validation.

**Данные:** 768 val-файлов → **745** с ≥1 импульсом после NABat quality filter → **6392** импульса, **26** классов  
**Split:** 85/15, stratify, `random_state=42`  
**Модели:** `cnn_bat_a0_best.pt` (ep.37), `resnet18_bat_best.pt` (ep.14), `vit_lfpe_bat_finetune_best_super.pt` (ViT no SSL, ep.8), `vit_lfpe_bat_finetune_best.pt` (ViT + SSL, ep.35)

---

## 1. Общие метрики

| Метрика | CNN | ResNet | ViT no SSL | ViT + SSL | NABat ML (статья) |
|---------|-----|--------|------------|-----------|-------------------|
| **Pulse-level accuracy** | **85.0%** | **85.2%** | 79.8% | 81.6% | 83% (validation) |
| Pulse-level weighted precision | 85.2% | 85.6% | 80.4% | 81.8% | 80% (validation) |
| **Pulse-level macro-F1** | **0.842** | **0.841** | 0.788 | 0.806 | — |
| Pulse-level weighted F1 | 0.849 | 0.852 | 0.798 | 0.815 | — |
| | | | | | |
| **File-level accuracy** (mean softmax по импульсам) | **90.5%** | **90.0%** | 87.8% | 88.3% | **92%** (test + range maps) |
| File-level weighted precision | 91.1% | 90.3% | 88.6% | 88.9% | 92% |
| File-level macro-F1 | 0.904 | 0.900 | 0.879 | 0.883 | — |
| File-level majority vote accuracy | 88.9% | 89.0% | 86.7% | 88.1% | — |
| | | | | | |
| File-level + conf ≥ 0.57* | 96.4% (609/745) | 95.2% (626/745) | 93.5% (604/745) | 92.2% (644/745) | часть их pipeline |
| Отброшено как NoID (conf < 0.57) | 136 файлов | 119 файлов | 141 файлов | 101 файлов | — |
| | | | | | |
| Согласие предсказаний CNN ↔ ResNet | — | **90.0%** | — | — | — |
| Разница macro-F1 (best ckpt) | 0.8419 | 0.8411 | 0.7884 | 0.8064 | Δ CNN−ResNet = **0.0008** |

\* conf threshold из статьи Khalighifar et al. 2022, **без range maps** — метрика завышена, т.к. сложные файлы отфильтровываются как NoID.

---

## 2. Уровни оценки

| Уровень | Что считается | CNN | ResNet | ViT no SSL | ViT + SSL | Зачем |
|---------|---------------|-----|--------|------------|-----------|-------|
| **Pulse-level** | каждый импульс = 1 пример | 85% acc, 0.84 macro-F1 | 85% acc, 0.84 macro-F1 | 80% acc, 0.79 macro-F1 | 82% acc, 0.81 macro-F1 | как validation в NABat ML |
| **File-level** | среднее softmax → 1 label на файл | **90.5% acc** | 90.0% acc | 87.8% acc | 88.3% acc | как test в NABat ML |
| **Macro-F1** | среднее F1 по классам (без весов) | 0.842 | 0.841 | 0.788 | 0.806 | честно к сложным видам |
| **Weighted accuracy** | = overall accuracy при single-label | ≈ acc | ≈ acc | ≈ acc | ≈ acc | метрика из статьи |

---

## 3. Per-class file-level identification rate (CNN, mean prob)

| Класс | ID rate | Файлов | | Класс | ID rate | Файлов |
|-------|---------|--------|---|-------|---------|--------|
| **MYEV** | **100%** | 30/30 | | **MYTH** | **100%** | 30/30 |
| COTO | 96.0% | 24/25 | | EUMA | 96.7% | 29/30 |
| LAIN | 96.7% | 29/30 | | MYCI | 96.7% | 29/30 |
| LANO | 96.4% | 27/28 | | LASE | 96.4% | 27/28 |
| MYGR | 96.4% | 27/28 | | NYMA | 96.4% | 27/28 |
| MYCA | 93.3% | 28/30 | | MYLU | 93.3% | 28/30 |
| MYSE | 93.3% | 28/30 | | MYSO | 93.3% | 28/30 |
| MYVO | 93.3% | 28/30 | | **PAHE** | **93.1%** | 27/29 |
| ANPA | 90.0% | 27/30 | | MYYU | 90.0% | 27/30 |
| NOISE | 90.0% | 18/20 | | TABR | 85.2% | 23/27 |
| LACI | 83.3% | 25/30 | | PESU | 82.8% | 24/29 |
| EPFU | 80.0% | 24/30 | | LABO | 79.3% | 23/29 |
| **MYLE** | **70.0%** | 21/30 | | **LABL** | **66.7%** | 16/24 |

**Классов с ≥90% file ID rate:** **19 / 26** (в статье: 19–20 / 31)

### 3b. Per-class file-level identification rate (ViT, mean prob)

| Класс | ViT no SSL | ViT + SSL | Файлов |
|-------|------------|-----------|--------|
| MYEV | 96.7% | **100%** | 29–30/30 |
| MYTH | **100%** | **100%** | 30/30 |
| EUMA | 96.7% | 96.7% | 29/30 |
| LAIN | **100%** | 96.7% | 29–30/30 |
| MYGR | 96.4% | 96.4% | 27/28 |
| NYMA | **100%** | 96.4% | 27–28/28 |
| COTO | 92.0% | 96.0% | 23–24/25 |
| MYCI | 86.7% | 93.3% | 26–28/30 |
| MYSO | 86.7% | 93.3% | 26–28/30 |
| PAHE | 89.7% | 93.1% | 26–27/29 |
| LASE | **100%** | 92.9% | 26–28/28 |
| MYVO | 80.0% | 90.0% | 24–27/30 |
| LABO | 89.7% | 89.7% | 26/29 |
| LANO | 78.6% | 89.3% | 22–25/28 |
| ANPA | 86.7% | 86.7% | 26/30 |
| MYLU | 80.0% | 86.7% | 24–26/30 |
| MYSE | 93.3% | 86.7% | 26–28/30 |
| MYYU | 86.7% | 86.7% | 26/30 |
| TABR | 88.9% | 85.2% | 23–24/27 |
| NOISE | 90.0% | 80.0% | 16–18/20 |
| PESU | 86.2% | 79.3% | 23–25/29 |
| LABL | 79.2% | 79.2% | 19/24 |
| LACI | 76.7% | 76.7% | 23/30 |
| MYCA | 83.3% | 76.7% | 23–25/30 |
| MYLE | 66.7% | 76.7% | 20–23/30 |
| EPFU | 73.3% | 70.0% | 21–22/30 |

**Классов с ≥90% file ID rate:** ViT no SSL **10 / 26**, ViT + SSL **12 / 26**

---

## 4. Главные путаницы (pulse-level)

| Пара (true → pred) | CNN | ResNet | ViT no SSL | ViT + SSL |
|--------------------|-----|--------|------------|-----------|
| LABL → LABO | **41** (20%) | 22 (11%) | 32 (16%) | — |
| MYLU → MYVO | 34 (12%) | 39 (13%) | 42 (15%) | 40 (14%) |
| MYLE → MYSE | 27 (11%) | 28 (11%) | 35 (14%) | 26 (11%) |
| MYYU → MYCA | 29 (10%) | 29 (10%) | — | 29 (10%) |
| MYCA → MYYU | — | — | 37 (14%) | **50 (20%)** |
| MYSE → MYEV | 30 (8%) | 20 (6%) | 27 (8%) | 35 (10%) |
| EPFU → ANPA | — | — | 34 (10%) | 37 (11%) |
| LANO → TABR | — | — | 33 (16%) | 25 (12%) |
| EPFU → LANO | 22 (6%) | 20 (6%) | — | — |

---

## 5. Per-class F1: где CNN и ResNet расходятся

| Класс | CNN F1 | ResNet F1 | Δ (CNN−ResNet) |
|-------|--------|-----------|----------------|
| NOISE | 0.764 | 0.695 | **+0.069** |
| LACI | 0.715 | 0.684 | +0.031 |
| EPFU | 0.822 | 0.852 | −0.030 |
| MYSE | 0.846 | 0.875 | −0.029 |
| MYTH | 0.944 | 0.964 | −0.020 |
| MYCI | 0.814 | 0.791 | +0.023 |

### 5b. Per-class F1: где ViT no SSL и ViT + SSL расходятся

| Класс | ViT no SSL F1 | ViT + SSL F1 | Δ (SSL−no SSL) |
|-------|---------------|--------------|----------------|
| MYCI | 0.710 | 0.813 | **+0.103** |
| NOISE | 0.667 | 0.745 | **+0.079** |
| MYSO | 0.858 | 0.933 | +0.075 |
| LABL | 0.646 | 0.696 | +0.049 |
| MYVO | 0.712 | 0.759 | +0.047 |
| MYLE | 0.752 | 0.798 | +0.047 |
| MYYU | 0.834 | 0.789 | −0.045 |
| MYLU | 0.693 | 0.737 | +0.044 |
| EPFU | 0.701 | 0.743 | +0.042 |
| LABO | 0.760 | 0.801 | +0.041 |

---

## 6. Обучение и артефакты

| | CNN | ResNet | ViT no SSL | ViT + SSL |
|---|-----|--------|------------|-----------|
| Best epoch | 37 / 40 | 14 / 24 (early stop) | 8 / 21 | 35 / 40 |
| Best macro-F1 (log) | 0.8419 | 0.8411 | 0.7884 | 0.8064 |
| train_loss / val_loss (≈best) | 0.65 / 1.11 | 0.64 / 1.11 | 0.72 / 1.31 | 0.63 / 1.42 |
| Confusion matrix | `cnn_bat_a0_confusion_matrix.png` | `resnet18_bat_confusion_matrix.png` | — | `vit_lfpe_bat_confusion_matrix.png` |
| Checkpoint | `cnn_bat_a0_best.pt` | `resnet18_bat_best.pt` | `vit_lfpe_bat_finetune_best_super.pt` | `vit_lfpe_bat_finetune_best.pt` |
| Lightning logs | `lightning_logs/cnn_baseline/` | `lightning_logs/resnet_baseline/` | `lightning_logs/finetune1/` | `lightning_logs/finetune/` |

---

## 7. Сравнение со статьей NABat ML

| | NABat ML (Khalighifar et al. 2022) | Этот проект (CNN, val) |
|---|-------------------------------------|-------------------------|
| Pulse-level accuracy | 83% | **85%** |
| File-level weighted accuracy | 92% (+ range maps) | **90.5%** (без range maps) |
| Классов с ≥90% file ID rate | 19–20 / 31 | **19 / 26** |
| PAHE (per-species) | 99% (их test) | 93.1% (27/29 файлов) |
| LABL (per-species) | 53% (их test) | 66.7% (16/24) |

**Про 0.99:** в статье это identification rate **отдельного вида** (PAHE), не overall accuracy модели.

---

## 8. OOD: CNN на файлах вне subset_200 (full `cleaned`)

**Протокол:** inference только, модель `cnn_bat_a0_best.pt` (обучена на `cleaned_subset_200`).  
**OOD pool:** 12 006 файлов из `cleaned/` (17 122 total), которых **не было** в subset_200.  
**Eval sample:** 768 файлов, stratified, `seed=43` (сопоставимо с val subset).  
**Скрипт:** `scripts/eval_cnn_ood.py --sample 768 --seed 43`

| Метрика | Subset val (in-distribution) | OOD (full \ subset_200) | Δ |
|---------|-------------------------------|-------------------------|---|
| Файлов / импульсов | 745 / 6392 | 746 / 6208 | — |
| **Pulse-level accuracy** | **85.0%** | **79.1%** | **−5.9 pp** |
| Pulse-level weighted precision | 85.2% | 81.8% | −3.4 pp |
| **Pulse-level macro-F1** | **0.842** | **0.681** | **−0.161** |
| **File-level accuracy** | **90.5%** | **83.8%** | **−6.7 pp** |
| File-level macro-F1 | 0.904 | 0.792 | −0.112 |
| File-level + conf ≥ 0.57 | 96.4% (609/745) | 96.0% (501/746) | NoID: 245 vs 136 |

**Ограничения OOD eval:**
- Классы **COTO, LABL, NYMA** полностью в subset_200 → в OOD их **нет** (23/26 species).
- Macro-F1 на OOD считается по присутствующим классам и **занижен** относительно subset val.
- Полный OOD pool — 12 006 файлов; для полного прогона: `uv run python scripts/eval_cnn_ood.py` (без `--sample`).

**Вывод:** на **новых файлах** из полного датасета CNN заметно проседает (−6–7 pp file-level acc, macro-F1 −0.16 pulse-level) — модель переобучена на конкретную подвыборку 200/класс, а не на весь `cleaned`.

---

## 9. Итог

| Вопрос | Ответ |
|--------|-------|
| CNN vs ResNet? | Практически равны (0.842 vs 0.841 macro-F1) |
| ViT no SSL vs ViT + SSL? | SSL даёт **+0.018** macro-F1 (0.788 → 0.806), file acc **+0.5 pp** (87.8% → 88.3%) |
| ViT vs CNN/ResNet? | ViT + SSL ниже baselines: **−0.036** macro-F1 vs CNN, file acc **−2.2 pp** |
| Плохие 0.84 macro-F1? | На file-level accuracy **~90.5%** in-distribution — близко к **92%** в статье |
| OOD generalization? | **File acc 83.8%**, macro-F1 **0.68** pulse-level — заметный drop |
| SSL поможет? | На subset val SSL поднял ViT, но до CNN/ResNet не дотягивает |
| Узкое место | Myotis-кластер, LABL/LABO, NOISE + domain shift между subset и full |

---

## 10. Честное сравнение ViT + SSL (variant A)

**Протокол:** SSL pretrain и supervised fine-tune на **одном** `cleaned_subset_200`, split 85/15, `seed=42`, NABat v2 (3×100×100).

| Этап | Данные | Скрипт |
|------|--------|--------|
| SSL pretrain | train+val pulses subset (~34k / ~6.4k) | `uv run python vit_lfpe_ssl_pretrain.py` |
| ViT fine-tune + SSL | train subset, balanced sampler | `uv run python vit_lfpe_bat_baseline.py` |
| ViT fine-tune без SSL | то же | `uv run python vit_lfpe_bat_baseline.py --no-ssl` |
| Baselines | CNN / ResNet | `supervised_*_baseline.py` |

**NABat v2 в SSL:** `load_spec` → NABat pipeline, ViT `PATCH_SIZE=(10,10)`, `SPEC_CHANNELS=3`, MAE `patch_dim=300`, recon demo RGB.

**Старый `vit_lfpe_ssl_best.pt` несовместим** (log-STFT 1×128×256) — нужен fresh pretrain.

Ablation на full cleaned: `vit_lfpe_ssl_pretrain.py --full` (нечестно vs subset CNN).

**Результаты ViT на том же val (inference, 2026-07-09):**

| Метрика | ViT no SSL | ViT + SSL | Δ (SSL − no SSL) |
|---------|------------|-----------|------------------|
| Pulse-level macro-F1 | 0.788 | 0.806 | **+0.018** |
| Pulse-level accuracy | 79.8% | 81.6% | +1.8 pp |
| File-level accuracy | 87.8% | 88.3% | +0.5 pp |
| File-level macro-F1 | 0.879 | 0.883 | +0.004 |

---

## Протокол подсчёта file-level метрик

1. Для каждого WAV: inference на всех импульсах, прошедших NABat quality filter.
2. **Mean probability:** усреднение softmax-векторов по импульсам → argmax.
3. **Majority vote:** наиболее частый класс среди импульсов.
4. **Conf ≥ 0.57:** файлы с max(mean_prob) < 0.57 исключаются как NoID (как condition 2 в статье, без range maps).

Ссылка на статью: [Khalighifar et al., 2022](https://doi.org/10.1111/1365-2664.14280)
