# Классификация эхолокационных сигналов летучих мышей

Проект посвящён классификации видов летучих мышей по коротким фрагментам ультразвуковых записей. Эхолокационные импульсы обладают выраженной частотно-временной структурой (FM-модуляция, гармоники, различия по длительности и полосе частот), что делает задачу пригодной для методов глубокого обучения на спектральных представлениях.

Используется подвыборка `cleaned_subset_200` — 26 классов, метаданные в `cleaned_subset_200/audio_metadata_cleaned.csv`.

## Протокол экспериментов

Все модели обучаются и оцениваются в единых условиях:

- разбиение train/validation: 85/15, `stratify`, `random_state=42`;
- длина клипа: 0.5 с при частоте дискретизации 192 kHz;
- основная метрика: macro-F1;
- дополнительная метрика: balanced accuracy.

Validation содержит 768 записей.

## Экспериментальные контуры

Сравниваются три независимых подхода.

**A0 — CNN с учителем.** Baseline на log-STFT без предобучения. Ноутбук: `supervised_cnn_baseline.ipynb`.

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
| B2 | BEATs + uniform FB | **0.652** | 0.657 | `checkpoints/beats_bat_finetune_best.pt` |
| A0 | CNN + log-STFT | 0.466 | 0.481 | `checkpoints/cnn_baseline_groupnorm_finetune_best.pt` |
| B0 | AudioMAE | 0.181 | 0.230 | `checkpoints/audiomae_bat_finetune_best.pt` |

BEATs (B2) превосходит CNN (A0) на 0.186 macro-F1 при полном цикле дообучения (лучшая эпоха — 35). Вероятная причина — сочетание предобученного энкодера и согласованного частотного представления на исходной частоте дискретизации.

Результат AudioMAE (B0) получен после 4 эпох и не является сопоставимым с завершёнными прогонами A0 и B2; требуется дообучение до заданного бюджета эпох.

На поздних эпохах BEATs наблюдается рост разрыва между train и validation loss. Для оценки и inference используется чекпоинт с максимальным macro-F1, а не последняя эпоха. Лог метрик: `checkpoints/beats_bat_train_log.txt`.

## Пайплайн BEATs (B2)

```
waveform (192 kHz) → STFT → 128 равномерных частотных полос → [T, 128]
    → BEATs_iter3 encoder → mean pooling → linear classifier (26 классов)
```

Код: `beats_bat.py`, ноутбук: `beats_ultrasound_bat_baseline.ipynb`.

При возобновлении обучения из `beats_bat_finetune_best.pt` загружается поле `classifier_state`; архитектура backbone инициализируется из `BEATs_iter3.pt`. Функция `load_beats_checkpoint` предназначена для официального pretrained-файла и не совместима с finetune-чекпоинтом.

## Дальнейшие шаги

1. Завершить дообучение AudioMAE (B0) до 40 эпох.
2. Реализовать и оценить контур C (masked AE, сравнение LF-PE и sin-cos).
3. Провести абляции частотной предобработки: PCEN, полосовая фильтрация.

Разведочный анализ данных: `EDA_local.ipynb`.

## Ссылки

- Chen et al., [BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058)
- Huang et al., [Masked Autoencoders that Listen](https://arxiv.org/abs/2203.16609)
- Li et al., [Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding](https://arxiv.org/abs/2106.02795)
- Song et al., [MADUV](https://www.isca-archive.org/interspeech_2025/song25_interspeech.pdf) — uniform filter bank для ультразвукового ввода
