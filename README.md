# Классификация эхолокационных сигналов летучих мышей

Проект посвящён классификации видов летучих мышей по коротким фрагментам ультразвуковых записей. Эхолокационные импульсы обладают выраженной частотно-временной структурой (FM-модуляция, гармоники, различия по длительности и полосе частот), что делает задачу пригодной для методов глубокого обучения на спектральных представлениях.

Используется подвыборка `cleaned_subset_200` — ~5116 записей, 26 классов, метаданные в `cleaned_subset_200/audio_metadata_cleaned.csv`.

## Протокол экспериментов

Supervised-бейзлайны A0, B1 и B2 обучаются и оцениваются в единых условиях:

- разбиение train/validation: 85/15, `stratify`, `random_state=42`;
- частота дискретизации: 192 kHz;
- из каждой записи берётся фрагмент **2 с**: на train — случайный или смещённый к участку с большей энергией (в 70% случаев, из 8 кандидатов), на val — центральный;
- балансировка классов: `WeightedRandomSampler`;
- основная метрика: macro-F1;
- дополнительная метрика: balanced accuracy.

Дальше пайплайны расходятся:

**A0 и B1 (log-STFT).** Спектрограмма строится по полосе 5–96 kHz, приводится к размеру 128×256 и нормализуется по каждому примеру. На train добавляется SpecAugment: маски по частоте и времени, случайное изменение громкости (±6 dB).

**B2 (BEATs).** Тот же двухсекундный фрагмент, но вместо log-STFT — равномерный filter bank на исходных 192 kHz: STFT, 128 частотных полос, нормализация под BEATs. Временной контекст не обрезается (~751 кадр). На train — лёгкий jitter громкости waveform (±6 dB).

Validation содержит 768 записей.

**B0 (AudioMAE)** пока обучается на прежнем протоколе (клип 0.5 с, ресемпл до 16 kHz) и напрямую не сопоставим с A0/B1/B2.

## Пайплайн BEATs (B2)

1. Берём 2-секундный фрагмент на 192 kHz (crop — как в общем протоколе).
2. Вычитаем среднее по waveform, считаем STFT (окно 2048, hop 512).
3. Усредняем амплитуду по 128 равномерным частотным полосам и нормализуем так, как ожидает BEATs.
4. Получаем матрицу примерно 751×128 — без урезания по времени.
5. Прогоняем через энкодер BEATs_iter3, усредняем по времени, классифицируем на 26 видов.

Код: `beats_bat.py`, ноутбук: `beats_ultrasound_bat_baseline.ipynb`.

## Экспериментальные контуры

Сравниваются несколько независимых подходов.

**A0 — CNN с учителем.** Baseline `BatCNNA0` на log-STFT (5–96 kHz, 128×256) без предобучения (~1M параметров). Ноутбук: `supervised_cnn_baseline.ipynb`.

**B1 — ResNet18 с учителем.** Усиленный supervised-baseline на том же log-STFT-пайплайне. Ноутбук: `supervised_resnet_baseline.ipynb`.

**B0, B2 — перенос предобученных аудио-моделей.** Энкодеры, обученные на AudioSet, дообучаются на целевом домене:
- **B0 (AudioMAE):** Kaldi fbank, ресемплинг до 16 kHz. Ноутбук: `audiomae_finetune_baseline.ipynb`;
- **B2 (BEATs):** uniform filter bank на 192 kHz без даунсэмплинга, без обрезки по времени ([MADUV, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/song25_interspeech.pdf)). Ноутбук: `beats_ultrasound_bat_baseline.ipynb`.

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
| A0 | CNN A0 + log-STFT | 0.788 | 0.790 | `checkpoints/cnn_bat_a0_best.pt` |
| B0 | AudioMAE | 0.340 | 0.360 | `checkpoints/audiomae_bat_finetune_best.pt` |
| B1 | ResNet18 + log-STFT | **0.818** | 0.818 | `checkpoints/resnet18_bat_best.pt` |
| B2 | BEATs + uniform FB | **0.803** | 0.806 | `checkpoints/beats_bat_finetune_best.pt` |

Лучший результат — ResNet18 (B1), macro-F1 0.818 (эпоха 40). BEATs (B2) — 0.803 (эпоха 37), CNN A0 — 0.788 (эпоха 40).

AudioMAE (B0) — 0.340 (эпоха 28, early stopping на 38); уступает A0, B1 и B2. Лог метрик: `checkpoints/audiomae_bat_train_log.txt`.

На поздних эпохах BEATs наблюдается рост разрыва между train и validation loss. Для оценки и inference используется чекпоинт с максимальным macro-F1, а не последняя эпоха. Лог метрик: `checkpoints/beats_bat_train_log.txt`.

## Дальнейшие шаги

1. **B2 (BEATs):** resume с абляциями LR и регуляризации; смягчить переобучение (рост разрыва train/val loss на поздних эпохах).
2. **C:** pretrain masked autoencoder на log-STFT и сравнение LF-PE vs sin-cos на downstream-классификации (`frequency_aware_ssl_bat_pipeline.ipynb`).

Разведочный анализ данных: `EDA_local.ipynb`.

## Ссылки

- Chen et al., [BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058)
- Huang et al., [Masked Autoencoders that Listen](https://arxiv.org/abs/2203.16609)
- Li et al., [Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding](https://arxiv.org/abs/2106.02795)
- Song et al., [MADUV](https://www.isca-archive.org/interspeech_2025/song25_interspeech.pdf) — uniform filter bank для ультразвукового ввода
