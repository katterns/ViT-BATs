# Классификация эхолокационных сигналов летучих мышей

Проект посвящён классификации видов летучих мышей по коротким фрагментам ультразвуковых записей. Эхолокационные импульсы обладают выраженной частотно-временной структурой (FM-модуляция, гармоники, различия по длительности и полосе частот), что делает задачу пригодной для методов глубокого обучения на спектральных представлениях.

Используется подвыборка `cleaned_subset_200` — ~5116 записей, 26 классов, метаданные в `cleaned_subset_200/audio_metadata_cleaned.csv`.

## Протокол экспериментов

Supervised-бейзлайны A0, B1 и B2 обучаются и оцениваются в единых условиях:

- разбиение train/validation: 85/15, `stratify`, `random_state=42`;
- частота дискретизации: 192 kHz;
- длина клипа: **2 с** (train: random / energy-biased crop; val: center crop);
- балансировка классов: `WeightedRandomSampler`;
- основная метрика: macro-F1;
- дополнительная метрика: balanced accuracy.

Validation содержит 768 записей.

**B0 (AudioMAE)** пока обучается на прежнем протоколе (клип 0.5 с, ресемпл до 16 kHz) и напрямую не сопоставим с A0/B1/B2.

## Экспериментальные контуры

Сравниваются несколько независимых подходов.

**A0 — CNN с учителем.** Baseline `BatCNNA0` на log-STFT без предобучения (~1M параметров). Ноутбук: `supervised_cnn_baseline.ipynb`.

**B1 — ResNet18 с учителем.** Усиленный supervised-baseline на том же log-STFT-пайплайне. Ноутбук: `supervised_resnet_baseline.ipynb`.

**B0, B2 — перенос предобученных аудио-моделей.** Энкодеры, обученные на AudioSet, дообучаются на целевом домене:
- **B0 (AudioMAE):** Kaldi fbank, ресемплинг до 16 kHz — `audiomae_finetune_baseline.ipynb`;
- **B2 (BEATs):** uniform filter bank на 192 kHz без даунсэмплинга ([MADUV, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/song25_interspeech.pdf)) — `beats_ultrasound_bat_baseline.ipynb`.

**C — собственный ViT с masked autoencoder.** Log-STFT, абляция positional encoding: Learnable Fourier PE и sin-cos. Реализация в `frequency_aware_ssl_bat_pipeline.ipynb`.

Предобученные веса:
```bash
python scripts/download_beats_pretrained.py    # checkpoints/BEATs_iter3.pt
python scripts/download_audiomae_pretrained.py # checkpoints/audiomae_pretrained.pth
```

## Результаты

Лучшие чекпоинты на validation:

| Контур | Модель | macro-F1 | bal. acc. | Чекпоинт |
|--------|--------|----------|-----------|----------|
| B1 | ResNet18 + log-STFT | **0.818** | 0.818 | `checkpoints/resnet18_bat_best.pt` |
| A0 | CNN A0 + log-STFT | **0.788** | 0.790 | `checkpoints/cnn_bat_a0_best.pt` |
| B2 | BEATs + uniform FB | 0.652 | 0.657 | `checkpoints/beats_bat_finetune_best.pt` |
| B0 | AudioMAE | 0.181 | 0.230 | `checkpoints/audiomae_bat_finetune_best.pt` |

Supervised-модели B1 и A0 превосходят BEATs (B2) на validation. Лучший результат — ResNet18 (B1), macro-F1 0.818 (эпоха 40); CNN A0 — 0.788 (эпоха 40). BEATs (B2) — 0.652 (лучшая эпоха — 37).

Результат AudioMAE (B0) получен после 4 эпох и не является сопоставимым с завершёнными прогонами A0, B1 и B2; требуется дообучение до заданного бюджета эпох и перенос на общий протокол.

На поздних эпохах BEATs наблюдается рост разрыва между train и validation loss. Для оценки и inference используется чекпоинт с максимальным macro-F1, а не последняя эпоха. Лог метрик: `checkpoints/beats_bat_train_log.txt`.

## Пайплайн BEATs (B2)

```
waveform (192 kHz, 2 с) → STFT → 128 равномерных частотных полос → [T, 128]
    → BEATs_iter3 encoder → mean pooling → linear classifier (26 классов)
```

Код: `beats_bat.py`, ноутбук: `beats_ultrasound_bat_baseline.ipynb`.

## Дальнейшие шаги

1. Завершить дообучение AudioMAE (B0) до 40 эпох и перенести на общий протокол (2 с, 192 kHz).
2. Дообучить BEATs (B2) с resume и абляциями LR.
3. Реализовать и оценить контур C (masked AE, сравнение LF-PE и sin-cos).
4. Провести абляции частотной предобработки: PCEN, полосовая фильтрация.

Разведочный анализ данных: `EDA_local.ipynb`.

## Ссылки

- Chen et al., [BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058)
- Huang et al., [Masked Autoencoders that Listen](https://arxiv.org/abs/2203.16609)
- Li et al., [Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding](https://arxiv.org/abs/2106.02795)
- Song et al., [MADUV](https://www.isca-archive.org/interspeech_2025/song25_interspeech.pdf) — uniform filter bank для ультразвукового ввода
