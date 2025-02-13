import random

import torch
import torchaudio.transforms as T
from data_utils.augmentations.base import BatchAugmentation


class TimeMasking(BatchAugmentation):
    """Spectrogram Time-masking augmentations as defined by Hung. et. al. (2024)"""

    def __init__(
        self,
        n_masks: int = 2,
        max_mask_percent: float = 0.25,
        p: float = 1.0,
    ):
        """
        Initialize time masking augmentation with given parameters.

        Args:
            n_masks: Number of time mask blocks to apply
            max_mask_percent: Maximum percentage of time steps to mask (0.0-1.0)
            p: Probability of applying each mask
        """
        super().__init__()
        self.n_masks = n_masks
        self.max_mask_percent = max_mask_percent
        self.p = p
        self.time_masking = None  # Will be initialized in transform_audio

    def _initialize_time_mask(self, time_steps: int):
        """Initialize time masking transform with the maximum mask size."""
        max_mask_size = int(time_steps * self.max_mask_percent)
        self.time_masking = T.TimeMasking(
            time_mask_param=max_mask_size,
        )

    def apply_time_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time masking to the spectrogram."""
        # Initialize time masking if not done yet
        if self.time_masking is None:
            self._initialize_time_mask(spectrogram.size(-1))

        # Apply multiple masks
        masked_spec = spectrogram
        for _ in range(self.n_masks):
            if random.random() < self.p:
                masked_spec = self.time_masking(masked_spec)

        return masked_spec

    def transform_audio(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking to the input spectrogram.

        Args:
            spectrogram: Input spectrogram of shape (batch_size, freq_bins, time_steps)
                        or (freq_bins, time_steps)
        """
        return self.apply_time_mask(spectrogram)
