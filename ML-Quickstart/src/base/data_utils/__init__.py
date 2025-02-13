from abc import ABC, abstractmethod

import numpy as np
import torch

from .preprocess_files import preprocess  # noqa: F401

__all__ = ["preprocess", "DataSegment", "Processor", "zeropad"]


class DataSegment(ABC):
    """
    Abstract base class representing a segment of data.

    Methods:
        process_target: Processes the target data and performs any necessary augmentations.
        read_file: Reads a file and returns necessary contents.
    """

    @abstractmethod
    def return_sample(self, start_time: float, end_time: float) -> torch.Tensor:
        """
        Returns a sample of the data between the given start and end time.

        Args:
            start_time (float): The start time of the sample.
            end_time (float): The end time of the sample.

        Returns:
            Tensor(s): Requisite data from the resultant segment.
        """
        pass

    @staticmethod
    def read_file(filepath: str) -> torch.Tensor | np.ndarray | dict[str, torch.Tensor | np.ndarray]:
        pass


class Processor(ABC):
    """
    Abstract base class for processing audio, midi segments

    Methods:
        process: Processes input data and performs any necessary augmentations.
    """

    @abstractmethod
    def process(self):
        pass


def zeropad(x: torch.Tensor | np.ndarray, num: int) -> torch.Tensor | np.ndarray:
    """
    Right-pads NumPy array or PyTorch tensor with zeroes.

    Args:
        x (Union[torch.Tensor, np.ndarray]): The array to pad.
        num (int): The desired length of the padded array.

    Returns:
        Union[torch.Tensor, np.ndarray]: The padded array.

    """
    if len(x) >= num or num <= 0:
        return x
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 1:
            return torch.cat([x, torch.zeros((num - x.shape[0],))])
        else:
            return torch.cat([x, torch.zeros((num - x.shape[0], *x.shape[1:]))])
    elif len(x.shape) == 1:
        return np.concatenate([x, np.zeros((num - x.shape[0],))])
    else:
        return np.concatenate([x, np.zeros((num - x.shape[0], *x.shape[1:]))], axis=0)
