# ML-Quickstart : A general purpose template for (audio) ML Research

## Development Setup
1. Set up a virtual environment using either `venv` or `conda`, with Python version 3.10. Choose whichever virtual environment tool you're most comfortable with.

2. Install system-level dependencies, such as `ffmpeg` or `sox`.

3. If desired for readability, navigate to the project's root directory and run `pre-commit install`. This command configures pre-commit hooks that will automatically inspect and potentially rectify code issues before each commit, ensuring consistent code quality and style.

## Provided Modules
All modules are structured around pytorch-lightning and Hydra.
### Configs
Configs are stored on `ML-Quickstart/config/[module]`
### Base
- Contains basic structure and scaffolding to create a new task-specific module

#### Included Basic Components
1. `callbacks`
2. `constants` _(Overriden by Hydra, but included for base variables)_
3. `data_utils`:

   _Included Submodules : `augmentations`, `audio_utils`, `preprocessing`, basic torch audio dataset and dataloader_

   - `augmentations `

      _Includes base interfaces (all torch.nn.Module)_ :
      
      `BatchAugmentor`: for defining specific batch augmentations which can operate on both input audio and labels (e.g pitch shifting audio and output pitch distribution)

      `BatchAugmentation`: Similar to `torch.nn.Sequential`, chains a series of `BatchAugmentor` to create an augmentation chain

      
      _Definitions for pitch shifting, spectral augmentations at waveform level, and  time, frequency masking at spectrogram level are included_

4. `Inference`
5. `Models`

   _Included Submodules :_
   
    - `model_blocks`: all torch modules to define specific model logic 
    
    - `loss`: torch module to define losses 
    
    - `metrics`: torchmetrics class to define metrics 

   - `base`: LightningModule wrapper for training and inference



### Transcription
- Module for Training, Evaluation and Inference of MIDI-based music transcription. Partially based on https://github.com/bytedance/piano_transcription.

### TO:DO
- General Classification/Regression Module
- Recommendation Module
- Generative Module

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

A parser for MAESTRO metadata is already configured in the `transcription` module. If no metadata path is passed, files are randomly selected for train/test/val splits in an 80/10/10 fashion _(Configurable)_.

### Training
Run 'python ML-Quickstart/src/\[module\]/main.py to start training of a model. Ensure you run from the ML-Quickstart/ level for nested dependencies.

### Logging
#### Lightning Logger
Currently logging is only supported via tensorboard (configured in hydra). 

TODO: MLFlow/Wandb logging

#### Manifest/SNS

in `manifest_sns`, utils are included for 

1. Generation of a summary manifest json _(Currently logs output metrics, config, and model path)_

   - By default this will be placed in the default output directory per hydra (along with model artifacts and log stack trace) 
   - TODO: Artifact (plots, audio) Handling, SQL/DB integration

2. Notification of ~~anything~~ model training/evaluation results (inspired by AWS SNS service)

   - relevant args _(source, destination email, credentials. etc.)_ must be listed in manifest_sns/email_args.py. This will NOT be git tracked for security. This module will not run if email_args.py is being tracked by git and throw an exception. 

   - If you choose to use this, please create a burner source email to send messages instead of using a personal/enterprise/important one.

   - Please do not send spam to random people :)