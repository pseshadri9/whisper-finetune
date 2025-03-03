# whisper-finetune

## Development Setup
1. Set up a virtual environment using either `venv` or `conda`, with Python version 3.10. 

2. Install system-level dependencies, such as `ffmpeg` or `sox`.

3. If desired for readability, navigate to the project's root directory and run `pre-commit install`. This command configures pre-commit hooks that will automatically inspect and potentially rectify code issues before each commit, ensuring consistent code quality and style.

## Provided Modules
All modules are structured around pytorch-lightning and Hydra.
### Configs
Configs are stored on `ML-Quickstart/config/[module]`

## Usage
### Data Processing
1. Place data (or simlink to data) inside ML-Quickstart/data/raw 

_(location configurable through hydra for cloud/distributed training)_
   
2. update or create a new config in config/data/preprocess for each new dataset

Entire default file with all parameters listed below for clarity:
   
```
   _target_ : data_utils.preprocess
source_path: Null #path to directory containing all audio, label files
destination_path: data/preprocessed/
metadata: Null #path to metadata file if predefined, else will be generated on first run

audio_ext: .wav
label_ext: .midi
processed_ext: .npy

sample_rate: 16000
```

3. Update 'source_path' to the path to your dataset (Both audio and labels, edit 'audio_ext', 'label_ext' accordingly), and 'destination_path' to the desired path for pre-processed data (audio -> mono numpy arrays). File parser expects one label file for every audio file with paths to both in alphabetical order (else file-structure agnostic).
   
_currently, code accepts pairs of audio, label files out of the box for training; however, alignment must be assured before hand (pipeline will not check that each file is aligned with audio)._

_if label files are not needed (self-supervised/generative tasks), pass Null into the config. Dataloader will then return a Null value during training for labels_


4. If dataset contains a metadata file, pass path to 'metadata' and write a parser function in 'ML-Quickstart/src/\[module\]/data_utils/preprocess_files.py. 

### Training
Run 'python ML-Quickstart/src/whisper-finetune/main.py to start training of a model. Ensure you run from the ML-Quickstart/ level for nested dependencies.
