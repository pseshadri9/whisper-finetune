import json
import os
import sys

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from constants import AUDIO_EXT, DESTINATION_PATH, LABEL_EXT, PROCESSED_EXT, SAMPLE_RATE
from tqdm import tqdm

# from ray.experimental.tqdm_ray import tqdm

"""
Script to convert the audio from wav to 16Khz downsampled, 1-channel (mono) numpy arrays
for easy dataloading (memmap, etc.)

structure of source directory will be replicated in the destination directory

See relevant constants in constants/preprocess.py

Dependencies: torch, torchaudio, tqdm
"""


def preprocess(
    source_path: str | None = None,
    destination_path: str = DESTINATION_PATH,
    audio_ext: str = AUDIO_EXT,
    processed_ext: str = PROCESSED_EXT,
    sample_rate: int = SAMPLE_RATE,
    label_ext: str = LABEL_EXT,
    metadata: str | None = None,
) -> str:
    """
    Preprocesses audio files from a source directory and stores the processed files in a destination directory.
    Generate CSV storing a list of audio/label files for dataloading.

    Parameters:
        source_path (str): The path to the source directory containing the audio files.
        destination_path (str): The path to the destination directory where the processed files will be stored.
        audio_ext (str): The file extension of the audio files.
        processed_ext (str): The file extension of the processed audio files.
        sample_rate (int): The sample rate of the processed audio files.
        label_ext (str): The file extension of the label files.

    Returns:
        Tuple[str, str]: The paths to the audio and label csv files detailing the processed files.
    """
    # preprocess audio (in the interest of time/debugging, below is commented)
    #dir_name = preprocess_audio(source_path, destination_path, audio_ext, processed_ext, sample_rate)

    # For now, assume SOURCE_PATH contains labels (as it does for maestro)
    audio_files = os.path.join(source_path, "audio.csv")
    label_files = os.path.join(source_path, "labels.csv")
    generate_cache_file_list(source_path, processed_ext, filename=audio_files)
    generate_cache_file_list(source_path, label_ext, filename=label_files)

    # if maestro metadata present, generate data splits file from metadata
    if metadata:
        """
        Write custom parser if dataset contains metadata file, else generate
        e.g generate_maestro_data_splits_file(metadata, audio_files, label_files)
        """
        data_path = generate_data_splits_file(audio_files, label_files)
    else:
        data_path = generate_data_splits_file(audio_files, label_files)

    return data_path


def preprocess_audio(
    source_path: str, destination_path: str, audio_ext: str, processed_ext: str, sample_rate: int
) -> str:
    """
    Preprocesses audio files from a source directory and stores the processed files in a destination directory.

    Parameters:
        SOURCE_PATH (str): The path to the source directory containing the audio files.
        DESTINATION_PATH (str): The path to the destination directory where the processed files will be stored.
        AUDIO_EXT (str): The file extension of the audio files.
        PROCESSED_EXT (str): The file extension of the processed audio files.
        SAMPLE_RATE (int): The sample rate of the processed audio files.

    Returns:
        str: The path to the destination directory where the processed files are stored.
    """

    name = os.path.basename(os.path.normpath(source_path))

    destination_dir = os.path.join(destination_path, name)

    completed = set()

    """
    attempt to create cache dir for pre-processed files,
    if dir already exists, add all previously processed files to completed set
    """
    try:
        os.mkdir(destination_dir)
    except FileExistsError:
        completed = {
            f[: -len(processed_ext)]
            for _, _, files in os.walk(destination_dir)
            for f in files
            if f.endswith(processed_ext)
        }

    pbar = tqdm(os.walk(source_path), total=len(list(os.walk(source_path))))
    for root, dirs, files in pbar:
        prefix = os.path.basename(root[len(source_path) :])

        for x in dirs:
            os.makedirs(os.path.join(destination_dir, prefix, x), exist_ok=True)

        audio_files = sorted([x for x in files if x.endswith(audio_ext)])

        for idx, x in enumerate(audio_files):
            pbar.set_description(f"{round(idx / len(audio_files) * 100, 2)}%")

            # if file already processed, skip
            if x[: -len(audio_ext)] in completed:
                continue

            #waveform, sr = torchaudio.load(os.path.join(root, x))
            
            '''
            waveform = F.resample(waveform, orig_freq=sr, new_freq=sample_rate)

            waveform = torch.mean(waveform, axis=0).cpu().numpy()

            waveform = waveform.astype(np.float32)
            '''

            waveform = np.fromfile(os.path.join(root, x), dtype=np.int16)

            with open(os.path.join(root, f'{x[:-len(audio_ext)]}{processed_ext}'), "wb") as f:
                np.save(f, waveform)

    return source_path


def generate_cache_file_list(directory: str, extension: str, filename: str = "audio"):
    """
    Generate CSV storing a list of audio/label files for dataloading
    """
    if os.path.exists(filename + ".csv"):
        return

    cmd = 'find {0} -name "*{1}" | sort > {2}'
    os.system(cmd.format(directory, extension, filename))


def generate_maestro_data_splits_file(metadata_file: str, audio_files_csv: str, label_files_csv: str) -> str:
    """
    Parses maestro metadata for official train/val/test splits +
    generates a json file containing the processed file paths for each split
    """

    file_name = os.path.join(os.path.dirname(audio_files_csv), "data.json")
    if os.path.exists(file_name):
        return file_name

    with open(metadata_file) as f:
        data = json.load(f)

    train = set()
    val = set()
    test = set()

    for k, v in data["split"].items():
        if v == "train":
            train.add(os.path.basename(data["audio_filename"][k].split(".")[0]))
        elif v == "validation":
            val.add(os.path.basename(data["audio_filename"][k].split(".")[0]))
        elif v == "test":
            test.add(os.path.basename(data["audio_filename"][k].split(".")[0]))

    audio_files = np.loadtxt(audio_files_csv, dtype=str)
    labels = np.loadtxt(label_files_csv, dtype=str)
    names = [os.path.basename(os.path.normpath(x)).split(".")[0] for x in audio_files]

    train_files = list()
    val_files = list()
    test_files = list()

    train_labels = list()
    val_labels = list()
    test_labels = list()

    for x in range(len(names)):
        if names[x] in train:
            train_files.append(audio_files[x])
            train_labels.append(labels[x])
        elif names[x] in val:
            val_files.append(audio_files[x])
            val_labels.append(labels[x])
        elif names[x] in test:
            test_files.append(audio_files[x])
            test_labels.append(labels[x])

    for split, orig, audio, label in zip(
        ["train", "val", "test"],
        [train, val, test],
        [train_files, val_files, test_files],
        [train_labels, val_labels, test_labels],
    ):
        assert (
            len(orig) == len(audio) == len(label)
        ), f"{split} mismatch: {len(orig)} source files != {len(audio)} audio files != {len(label)} label files"

    _save_data_splits_file(file_name, train_files, val_files, test_files, train_labels, val_labels, test_labels)

    return file_name


def _save_data_splits_file(
    file_name: str,
    train_files: list[str],
    val_files: list[str],
    test_files: list[str],
    train_labels: list[str],
    val_labels: list[str],
    test_labels: list[str],
):
    data = dict()
    data["train"] = dict()
    data["val"] = dict()
    data["test"] = dict()

    data["train"]["audio"] = train_files
    data["train"]["labels"] = train_labels

    data["val"]["audio"] = val_files
    data["val"]["labels"] = val_labels

    data["test"]["audio"] = test_files
    data["test"]["labels"] = test_labels

    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)


def generate_data_splits_file(
    audio_files_csv: str, label_files_csv: str, proportion: list[float] = [0.8, 0.0, 0.2]
) -> str:
    audio_files = np.loadtxt(audio_files_csv, dtype=str)
    labels = np.loadtxt(label_files_csv, dtype=str)

    file_name = os.path.join(os.path.dirname(audio_files_csv), "data.json")
    if os.path.exists(file_name):
        return file_name

    # Create random permutation with given proportion [train, val, test]
    perm = np.random.permutation(len(audio_files))
    n = len(perm)
    train = perm[: int(n * proportion[0])]
    val = perm[int(n * proportion[0]) : int(n * (proportion[0] + proportion[1]))]
    test = perm[int(n * (proportion[0] + proportion[1])) :]

    train_files = audio_files[train].tolist()
    val_files = audio_files[val].tolist()
    test_files = audio_files[test].tolist()

    train_labels = labels[train].tolist()
    val_labels = labels[val].tolist()
    test_labels = labels[test].tolist()

    _save_data_splits_file(file_name, train_files, val_files, test_files, train_labels, val_labels, test_labels)

    return file_name


if __name__ == "__main__":
    if len(sys.argv) == 3:
        source_path, destination_path = tuple(sys.argv[1:])
    preprocess(source_path=source_path, destination_path=destination_path)
