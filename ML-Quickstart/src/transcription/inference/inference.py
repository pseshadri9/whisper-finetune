import json
import logging
from typing import Literal

import numpy as np
import torch
import torchaudio
from constants import SEGMENT_LENGTH, START_NOTE, VELOCITY_SCALE
from data_utils import zeropad
from data_utils.midi_utils import MIDI2Target, MIDIFile, Target2MIDI
from models import TranscriptionModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

log = logging.getLogger(__name__)

MODEL_OUTPUT_EXT = ".npy"


class Transcriptor:
    def __init__(
        self,
        model: TranscriptionModel,
        output_filetype: str = ".mid",
        batch_size: int = 16,
        segment_length: float = SEGMENT_LENGTH,
        device: Literal["cuda", "cpu"] = "cpu",
        p_threshold: float = 0.3,
        verbose=True,
    ):
        if not verbose:
            log.setLevel(logging.WARNING)

        self.model = model
        self.model.eval()

        self.sr = self.model.feature_extractor.sample_rate
        self.frames_per_second = self.sr / self.model.feature_extractor.hop_length
        self.n_classes = self.model.num_classes
        self.p_threshold = p_threshold
        self.segment_length = segment_length

        self.metadata = {
            "sr": self.sr,
            "frames_per_second": self.frames_per_second,
            "n_classes": self.n_classes,
            "p_threshold": p_threshold,
            "segment_length": segment_length,
        }

        self.midi_post_processor = Target2MIDI(
            self.frames_per_second,
            self.n_classes,
            START_NOTE,
            velocity_scale=VELOCITY_SCALE,
            onset_threshold=p_threshold,
            offset_threshold=p_threshold,
            frame_threshold=p_threshold,
            pedal_offset_threshold=p_threshold,
        )

        self.midi_pre_processor = MIDI2Target(
            self.segment_length,
            self.frames_per_second,
            START_NOTE,
            self.n_classes,
        )

        self.output_filetype = output_filetype
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Create post-processor
        self.model.eval()
        self.model.to(self.device)

    def transcribe(
        self,
        filepath: str,
        output_filename: str,
        progress_bar: bool = True,
        save_model_output: bool = False,
        output_metadata: bool = False,
    ) -> list[str]:
        log.info(f"Loading audio file from {filepath}...")
        waveform, sr = torchaudio.load(filepath)

        # Resample to 16000 Hz and convert to mono-channel
        waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        waveform = waveform.mean(dim=0)

        time = round(waveform.shape[0] / self.sr, 1)

        # Pad waveform if not evenly divisible by sample_rate * segment_length
        num_elem = self.sr * self.segment_length
        num_pad = int(num_elem - (waveform.shape[0] % num_elem))
        waveform = zeropad(waveform, waveform.shape[0] + num_pad)

        # Reshape waveform to tuple of elements (batch_size, sample_rate * segment_length)
        waveform = waveform.view(-1, int(self.sr * self.segment_length))
        waveform = torch.split(waveform, self.batch_size)
        num_batches = len(waveform)

        outputs = []

        if progress_bar:
            pbar = tqdm(range(num_batches))
        else:
            pbar = range(num_batches)

        log.info(f"Processing {time} s audio in {num_batches} batches...")
        # Process in batches
        for n_batch in pbar:
            # Get batch and transfer to device
            batch = waveform[n_batch].to(self.device)

            # Process batch
            with torch.no_grad():
                output = self.model(batch)

            # Save output in CPU
            outputs.append(self.model.to_device(output, cpu=True))

        # Concatenate all tensors and flatten batches for a combined output dict[str, tensor]
        total_output = {
            k: torch.cat([d[k].view(-1, self.n_classes) for d in outputs]).cpu().numpy() for k in outputs[0]
        }

        log.info("Converting to MIDI events...")
        midi_events = self.midi_post_processor.output_dict_to_midi_events(total_output)

        if not output_filename.endswith(self.output_filetype):
            output_filename = output_filename.split(".")[0] + self.output_filetype

        log.info(f"Saving to file {output_filename}...")
        self.midi_post_processor.write_events_to_midi(0, *midi_events, output_filename)

        files = [output_filename]

        # Save model output as separate numpy arrays
        if save_model_output:
            log.info("Saving model output...")
            outputs_prefix = output_filename.split(".")[0] + "_[{0}]" + MODEL_OUTPUT_EXT

            # Save mask_roll from processed midi for teacher-student training
            midi_file = MIDIFile(output_filename, self.midi_pre_processor)
            total_output["mask_roll"] = midi_file.return_sample(0, -1)["mask_roll"]

            for k, v in total_output.items():
                files.append(outputs_prefix.format(k.replace("output", "roll")))
                np.save(files[-1], v)

        if output_metadata:
            log.info("Saving metadata...")
            with open(output_filename.split(".")[0] + "_metadata.json", "w") as f:
                json.dump({"output_files": files, **self.metadata}, f, indent=4)

        return files
