import random

import torch
from data_utils.augmentations.base import BatchAugmentation
from torchaudio.functional import highpass_biquad, lowpass_biquad


class Highpass(torch.nn.Module):
    def __init__(self, cutoff_freq, sample_rate):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.sr = sample_rate

    def forward(self, audio):
        return highpass_biquad(audio, self.cutoff_freq, self.sr)


class Lowpass(torch.nn.Module):
    def __init__(self, cutoff_freq, sample_rate):
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.sr = sample_rate

    def forward(self, audio):
        return lowpass_biquad(audio, self.cutoff_freq, self.sr)


class SpectralAugmentations(BatchAugmentation):
    """Spectral augmentations as defined by Hung. et. al. (2024)"""

    def __init__(
        self,
        sample_rate: int = 16000,
        p: float = 0.2,
        snr_range: tuple[float, float] = (0.3, 0.5),
        gain_range: tuple[float, float] = (-10, 10),
        highpass_freq_range: tuple[int, int] = (20, 2000),
        lowpass_freq_range: tuple[int, int] = (4000, 8000),
    ):
        """
        Initialize audio augmentation pipeline with given parameters.

        Args:
            sample_rate: Audio sample rate in Hz
            p: Probability of applying each individual augmentation
            snr_range: Range of signal-to-noise ratio in dB for white noise
            gain_range: Range of random gain in dB
            highpass_freq_range: Range of highpass filter cutoff frequencies
            lowpass_freq_range: Range of lowpass filter cutoff frequencies
        """
        super().__init__()
        self.p = p
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.gain_range = gain_range
        self.highpass_freq_range = highpass_freq_range
        self.lowpass_freq_range = lowpass_freq_range

    def add_white_noise(self, audio: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add Gaussian white noise to the audio signal with specified SNR (dB)."""
        # Calculate the average signal power per sample
        signal_power = audio.norm(p=2).pow(2) / audio.numel()

        # Generate Gaussian white noise (mean=0, std=1)
        noise = torch.randn_like(audio)

        # Calculate noise power and desired SNR in linear scale
        noise_power = noise.norm(p=2).pow(2) / noise.numel()
        snr_linear = 10 ** (snr_db / 10)

        # Scale noise to achieve the target SNR
        scale = (signal_power / (noise_power * snr_linear)).sqrt()
        noisy_audio = audio + scale * noise

        return noisy_audio

    def apply_gain(self, audio: torch.Tensor, gain_db: float) -> torch.Tensor:
        """Apply random gain to the audio signal."""
        return audio * (10 ** (gain_db / 20))

    def apply_filters(self, audio: torch.Tensor, highpass_freq: int, lowpass_freq: int) -> torch.Tensor:
        """Apply high-pass and low-pass filters to the audio signal."""
        # High-pass filter
        highpass = Highpass(
            cutoff_freq=highpass_freq,
            sample_rate=self.sample_rate,
        )

        # Low-pass filter
        lowpass = Lowpass(
            cutoff_freq=lowpass_freq,
            sample_rate=self.sample_rate,
        )

        return lowpass(highpass(audio))

    def invert_polarity(self, audio: torch.Tensor) -> torch.Tensor:
        """Invert the polarity of the audio signal using torch.neg()."""
        return torch.neg(audio)

    def transform_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply each augmentation independently with probability p."""
        # Add white noise with probability p
        if random.random() < self.p:
            snr = random.uniform(*self.snr_range)
            audio = self.add_white_noise(audio, snr)

        # Apply random gain with probability p
        if random.random() < self.p:
            gain = random.uniform(*self.gain_range)
            audio = self.apply_gain(audio, gain)

        # Apply filters with probability p
        if random.random() < self.p:
            highpass_freq = random.randint(*self.highpass_freq_range)
            lowpass_freq = random.randint(*self.lowpass_freq_range)
            audio = self.apply_filters(audio, highpass_freq, lowpass_freq)

        # Invert polarity with probability p
        if random.random() < self.p:
            audio = self.invert_polarity(audio)

        return audio
