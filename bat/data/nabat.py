import colorsys
import math
from collections import namedtuple
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Metadata = namedtuple("Metadata", "offset, frequency, amplitude, time, snr, window")
Data = namedtuple("Data", "name, duration, sample_rate, metadata")

CLIP_MS = 50
WINDOW_OVERLAP = 0.008
MAX_FILE_MS = 45_000
MIN_FREQ_HZ = 5_000
MAX_FREQ_HZ = 100_000
SN_THRESH = 7.0
AMP_THRESH = 21.0
IMG_SIZE = 100
IMG_CHANNELS = 3


class Spectrogram:
    def __init__(self, overlap=0.008, sn_thresh=7, amp_thresh=21, window_length=50):
        self.colors = []
        self.img_height = 100
        self.img_width = 100
        self.img_channels = 3
        self.window_length = window_length
        self.maximum_file_length = 45000
        self.overlap = overlap * self.window_length
        self.sn_thresh = sn_thresh
        self.amp_thresh = amp_thresh
        self.fig = None

        for i in range(101):
            rgb = colorsys.hsv_to_rgb(i / 300.0, 1.0, 1.0)
            ll = [round(255 * x) for x in rgb]
            for j in range(len(ll)):
                ll[j] = ll[j] / 255
            self.colors.append(tuple(ll))

    def process_file(self, wav_file_name):
        try:
            sig, sr = librosa.load(wav_file_name, sr=None)
            duration = len(sig) / sr
            data = Data(wav_file_name, duration, sr, [])
        except Exception as e:
            print(e)
            return None

        for i in range(
            self.window_length,
            min(math.ceil((len(sig) / float(sr)) * 1000), self.maximum_file_length),
            int(self.window_length * (1 - self.overlap)),
        ):
            start = (i - self.window_length) / 1000
            end = i / 1000
            fsig = sig[int(start * sr) : int(end * sr)]
            metadata = self._process_window(fsig, sr, i)
            if metadata is not None:
                data.metadata.append(metadata)

        return data

    def _process_window(self, sig, sr, window_offset):
        root_size = int(0.001 * sr)
        hop_length = int(root_size / 4)

        stft_spec_window = librosa.stft(
            sig,
            n_fft=root_size,
            hop_length=hop_length,
            win_length=root_size,
            window="hamming",
        )
        stft_spec_window = np.abs(stft_spec_window) ** 2
        stft_spec_window = librosa.power_to_db(stft_spec_window)

        frequency_bands = librosa.fft_frequencies(sr=sr, n_fft=root_size)

        for i, b in enumerate(frequency_bands):
            if b <= 5000 or b >= min(100000, (sr / 2) - 2000):
                stft_spec_window[i] = [-500] * len(stft_spec_window[i])

        index = np.unravel_index(stft_spec_window.argmax(), stft_spec_window.shape)
        time_index = index[1]
        frequency_index = index[0]

        peak_frequency = frequency_bands[frequency_index]
        peak_time = time_index / 4

        if peak_time < self.window_length * 0.2 or peak_time > self.window_length * 0.8:
            return None
        if peak_frequency <= 5000 or peak_frequency >= min(100000, (sr / 2) - 2000):
            return None

        stft_spec_window = self._denoise_spec(stft_spec_window)

        freq_amp = stft_spec_window[frequency_index]
        r_other = np.sum(stft_spec_window) / (len(stft_spec_window) * len(stft_spec_window[0]))
        rsig = sum(freq_amp[time_index - 4 : time_index + 6]) / 10
        signal_noise_ratio = rsig / r_other
        amplitude = freq_amp[time_index]

        if signal_noise_ratio >= self.sn_thresh and amplitude >= self.amp_thresh:
            stft_spec_window = stft_spec_window.astype("float16")
            return Metadata(
                window_offset,
                peak_frequency,
                float(amplitude),
                peak_time,
                signal_noise_ratio,
                stft_spec_window,
            )
        return None

    def _get_Figure(self):
        if self.fig is None:
            self.fig = plt.figure(figsize=(1, 1), facecolor="black", dpi=100)
            self.ax = self.fig.add_axes([0, 0, 1, 1], facecolor="black")
            plt.margins(0)
        return self.fig, self.ax

    def make_spectrogram(self, sig, sr, low=5000, high=100000):
        try:
            root_size = int(0.001 * sr)
            hop_length = int(root_size / 4)

            fig, ax = self._get_Figure()
            ax.clear()

            librosa.display.specshow(
                sig,
                sr=sr,
                hop_length=hop_length,
                x_axis="s",
                y_axis="linear",
                ax=ax,
            )
            ax.set_ylim(low, high)
            ax.axis("off")

            img = self.fig2data(fig)
            img = np.array(img)
            img = img[..., :3].astype("float32")
            img /= 255.0
            return img
        except Exception as e:
            print(e)
            return None

    def fig2data(self, fig):
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf[..., :3]

    def _denoise_spec(self, spec):
        spec = spec - np.median(spec, axis=1, keepdims=True)
        spec = spec - np.median(spec, axis=0, keepdims=True)
        spec.clip(min=0, out=spec)
        return spec


_SPECTROGRAM = Spectrogram()


def process_file(path) -> Data | None:
    return _SPECTROGRAM.process_file(str(path))


def make_spectrogram_chw(window_spec: np.ndarray, sr: int) -> np.ndarray | None:
    """RGB float32 [3, H, W] для PyTorch."""
    img = _SPECTROGRAM.make_spectrogram(window_spec, sr)
    if img is None:
        return None
    return np.transpose(img, (2, 0, 1))


def metadata_for_offset(data: Data, window_offset: int) -> Metadata | None:
    for m in data.metadata:
        if m.offset == window_offset:
            return m
    return None
