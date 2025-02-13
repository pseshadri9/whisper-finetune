import json
import logging

import lightning as pl
import numpy as np
import torch
from base.data_utils.audio_utils import AudioProcessor, AudioSegment
from constants import BATCH_SIZE, NUM_WORKERS, SEED
from data_utils import zeropad
from midi_utils import MIDI2Target, MIDIFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

log = logging.getLogger(__name__)


class TranscriptionDataset(Dataset):
    def __init__(
        self,
        audio_files: list,
        label_files: list,
        sample_rate: int,
        num_classes: int,
        frames_per_second: int,
        segment_length: int,
        begin_note: int,
        random_seed: int = SEED,
    ):
        super().__init__()

        assert len(audio_files) == len(label_files)

        self.audio_files = audio_files
        self.label_files = label_files

        self.sr = sample_rate
        self.num_classes = num_classes
        self.frames_per_second = frames_per_second
        self.segment_length = segment_length
        self.begin_note = begin_note
        self.random_seed = random_seed

        self.audio_files, self.label_files = self.load_data()

    """
    Helper functions to convert between sample, time, and frame indices
    """

    def sample2time(self, X: int) -> float:
        return X / self.sr

    def time2sample(self, X: float) -> int:
        return int(round(X * self.sr))

    def time2frame(self, X: float) -> int:
        return int(round(X * self.frames_per_second))

    def frame2time(self, X: int) -> float:
        return X / self.frames_per_second

    def frame2sample(self, X: int) -> int:
        return self.time2sample(self.frame2time(X))

    def sample2frame(self, X: int) -> int:
        return self.time2frame(self.sample2time(X))

    def __len__(self):
        """
        Returns the total number of training examples in the dataset.

        Note that indices in this dataset are counting training samples.
        e.g (total_frames / segment_length) num samples, NOT frames, times, or audio samples

        Returns:
            int: The total number of training examples in the dataset.
        """
        total_frames = self.sample2frame(self.indices[-1])
        return int((total_frames - (self.segment_length * self.frames_per_second)) // self.segment_length)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns a sample of length segment_length starting at frame [item].

        Args:
            item (int): The index of the start frame of the sample

        Returns:
            tuple: A tuple containing the audio and label samples.
        """
        # Search for correct audio, midi file
        item *= self.segment_length
        start_sample = self.frame2sample(item)
        idx = np.searchsorted(self.indices, start_sample, side="right") - 1

        audio_file = self.audio_files[idx]
        label_file = self.label_files[idx]

        start_sample -= self.indices[idx]

        start_time = self.sample2time(start_sample)

        # Handle sample length less than segment length (end_time > EOF)
        if start_sample + self.time2sample(self.segment_length) > audio_file.num_samples():
            start_time_1 = start_time
            end_time_1 = self.sample2time(audio_file.num_samples())

            audio_1 = audio_file.return_sample(start_time_1, start_time_1 + self.segment_length)
            labels_1 = label_file.return_sample(start_time_1, end_time_1)

            # Zero-pad waveform and labels
            audio = audio_1
            label = {k: zeropad(labels_1[k], self.time2frame(self.segment_length)) for k in labels_1.keys()}

        else:
            audio = audio_file.return_sample(start_time, start_time + self.segment_length)
            label = label_file.return_sample(start_time, start_time + self.segment_length)

        return audio, label

    def load_data(self) -> tuple[list[AudioSegment], list[MIDIFile]]:
        audio_segments = []
        midi_files = []

        self.audio_processor = AudioProcessor(self.sr, seed=self.random_seed)
        self.midi_processor = MIDI2Target(
            self.segment_length, self.frames_per_second, self.begin_note, self.num_classes
        )

        for audio_file, label_file in tqdm(zip(self.audio_files, self.label_files), total=len(self.audio_files)):
            audio_segments.append(AudioSegment(audio_file, self.audio_processor, sr=self.sr))
            midi_files.append(MIDIFile(label_file, self.midi_processor))

        self.indices = np.cumsum([0] + [x.target.shape[0] for x in audio_segments])

        return audio_segments, midi_files


class TranscriptionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        metadata: str,
        sample_rate: int,
        num_classes: int,
        frames_per_second: int,
        segment_length: int,
        start_note: int,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        random_seed: int = SEED,
    ):
        super().__init__()
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.num_classes = num_classes
        self.frames_per_second = frames_per_second
        self.segment_length = segment_length
        self.start_note = start_note

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed

        # To ensure setup is only called once for TS training
        self.setup_called = False
        self.setup()
        self.setup_called = True

    def setup(self, stage=None):
        if self.setup_called:
            return
        with open(self.metadata) as f:
            data = json.load(f)

        self.train_audio_files = data["train"]["audio"]
        self.train_label_files = data["train"]["labels"]

        self.val_audio_files = data["val"]["audio"]
        self.val_label_files = data["val"]["labels"]

        self.test_audio_files = data["test"]["audio"]
        self.test_label_files = data["test"]["labels"]

        log.info(f"Instantiating Training set with {len(self.train_audio_files)} files")
        self.train_dataset = TranscriptionDataset(
            self.train_audio_files,
            self.train_label_files,
            self.sample_rate,
            self.num_classes,
            self.frames_per_second,
            self.segment_length,
            self.start_note,
            random_seed=self.random_seed,
        )

        log.info(f"Instantiating Validation set with {len(self.val_audio_files)} files")
        self.val_dataset = TranscriptionDataset(
            self.val_audio_files,
            self.val_label_files,
            self.sample_rate,
            self.num_classes,
            self.frames_per_second,
            self.segment_length,
            self.start_note,
            random_seed=self.random_seed,
        )

        log.info(f"Instantiating Test set with {len(self.test_audio_files)} files")
        self.test_dataset = TranscriptionDataset(
            self.test_audio_files,
            self.test_label_files,
            self.sample_rate,
            self.num_classes,
            self.frames_per_second,
            self.segment_length,
            self.start_note,
            random_seed=self.random_seed,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
