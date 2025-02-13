"""Default values for reference. Will be overwritten by config via hydra"""

import os

import torch

SEED = 1234

# data cache location
DESTINATION_PATH = "data/preprocessed/"

# Preprocessing args
AUDIO_EXT = ".wav"
LABEL_EXT = ".midi"
PROCESSED_EXT = ".npy"
SAMPLE_RATE = 16000

# MIDI args
NUM_CLASSES = 88  # Number of notes of piano
START_NOTE = 21  # MIDI note of A0, the lowest note of a piano.
SEGMENT_LENGTH = 10.0  # Training segment duration
FRAMES_PER_SECOND = 100
VELOCITY_SCALE = 128  # Used for post-processing model outputs to MIDI, To be implemented

# Data loader args
BATCH_SIZE = 16
NUM_WORKERS = os.cpu_count() // 2
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Mel spectrogram parameters
MSPEC_PARAMS = dict(
    n_fft=2048,
    win_length=2048,
    hop_length=SAMPLE_RATE // FRAMES_PER_SECOND,
    n_mels=229,
    f_min=30,
    f_max=SAMPLE_RATE // 2,
    center=True,
    pad_mode="reflect",
)
