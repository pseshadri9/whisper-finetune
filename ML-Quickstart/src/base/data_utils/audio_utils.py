import numpy as np
import sox
import torch
import torchaudio
from constants import PROCESSED_EXT, SAMPLE_RATE, SEED
from data_utils import DataSegment, Processor, zeropad


class Augmentor:
    """
    Waveform augmentor, see https://github.com/bytedance/piano_transcription/blob/master/utils/data_generator.py#L122
    """

    def __init__(self, sample_rate: int, seed=SEED):
        """Data augmentor."""

        self.sample_rate = sample_rate
        self.random_state = np.random.RandomState(seed)

    def augment(self, x):
        clip_samples = len(x)

        tfm = sox.Transformer()
        tfm.set_globals(verbosity=0)

        tfm.pitch(self.random_state.uniform(-0.1, 0.1, 1)[0])
        tfm.contrast(self.random_state.uniform(0, 100, 1)[0])

        tfm.equalizer(
            frequency=self.loguniform(32, 4096, 1)[0],
            width_q=self.random_state.uniform(1, 2, 1)[0],
            gain_db=self.random_state.uniform(-30, 10, 1)[0],
        )

        tfm.equalizer(
            frequency=self.loguniform(32, 4096, 1)[0],
            width_q=self.random_state.uniform(1, 2, 1)[0],
            gain_db=self.random_state.uniform(-30, 10, 1)[0],
        )

        tfm.reverb(reverberance=self.random_state.uniform(0, 70, 1)[0])

        aug_x = tfm.build_array(input_array=x, sample_rate_in=self.sample_rate)
        aug_x = zeropad(aug_x, clip_samples)[:clip_samples]  # pad/truncate

        return np.array(aug_x).astype(np.float32)

    def loguniform(self, low, high, size):
        return np.exp(self.random_state.uniform(np.log(low), np.log(high), size))


class AudioProcessor(Processor):
    def __init__(self, sample_rate: int, seed: int = SEED):
        self.sample_rate = sample_rate
        self.augmentor = Augmentor(sample_rate, seed=seed)

    def process(self, waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
        sample = self.augmentor.augment(waveform)
        sample = torch.from_numpy(sample)
        return sample


class AudioSegment(DataSegment):
    def __init__(
        self, filepath: str, processor: AudioProcessor, sr: int = SAMPLE_RATE, processed_ext: str = PROCESSED_EXT
    ) -> None:
        """
        Initializes an instance of the AudioSegment class.
        Loads and processes a segment of audio from an audio file.

        Args:
            filepath (str): The path to the audio file.
            start_time (float): The start time of the audio segment.
            processor: Audio Processor class to apply augmentations to the raw waveform
            sr (int): The desired sample rate of the audio file.
                - Numpy arrays will be inferred to be this rate
                - PyTorch tensors (from loading raw audio) will be converted to this rate

        Returns:
            None
        """
        self.sr = sr
        self.filepath = filepath
        self.processor = processor
        self.target = self.read_file(self.filepath, sr=sr, processed_ext=processed_ext)

    def num_samples(self) -> int:
        return self.target.shape[0]

    def num_seconds(self) -> float:
        return self.num_samples() / self.sr

    def return_sample(self, start_time: float, end_time: float) -> np.ndarray:
        """
        Returns a processed sample of audio between the given start and end time.

        Args:
            start_time (float): The start time of the sample.
            end_time (float): The end time of the sample.

        Returns:
            torch.Tensor: The audio segment processed by the audio processor.
        """
        if (end_time - start_time) * self.sr < 0:
            return torch.tensor([], dtype=torch.float32)

        if end_time >= self.num_seconds():
            arr = np.array(self.target[round(start_time * self.sr) :])
            sample = zeropad(arr, round(end_time - start_time) * self.sr)
        else:
            sample = self.target[round(start_time * self.sr) : round(end_time * self.sr)]
            sample = np.array(sample)

        """
        sox only operates on numpy arrays

        if (not isinstance(sample, torch.Tensor)):
            sample = torch.from_numpy(np.array(sample))
        """

        return self.processor.process(sample)

    @staticmethod
    def read_file(filepath: str, sr=SAMPLE_RATE, processed_ext=PROCESSED_EXT) -> torch.Tensor | np.ndarray:
        """
        Reads an audio file from the specified filepath and returns the audio segment.

        Args:
            filepath (str): The path to the audio file.
                - Can either be post-processed file of type 'processed_ext' or raw audio
            sr (int): The sample rate of the audio file.
                - Numpy arrays will be inferred to be this rate
                - raw audio will be resampled to this rate

            processed_ext (str): The file extension of the processed audio files.
                - defined in constants as '.npy''

        Returns:
            Union[torch.Tensor, np.ndarray]: The audio segment as a PyTorch tensor or a NumPy array.
        """

        if filepath.endswith(processed_ext):
            audio_segment = np.load(filepath, mmap_mode="r")
            assert len(audio_segment.shape) == 1, "Audio segment should be mono (shape (N_samples,)) if pre-processed"
            return audio_segment
        else:
            audio_segment, file_sr = torchaudio.load(filepath)  # Not memory mapped
            if len(audio_segment.shape) > 1:
                audio_segment = torch.mean(audio_segment, dim=0)  # convert to mono
            if file_sr != sr:
                audio_segment = torchaudio.functional.resample(audio_segment, file_sr, sr)
            assert len(audio_segment.shape) == 1, "Audio segment should be mono (shape (N_samples,)) if raw"

            return audio_segment
