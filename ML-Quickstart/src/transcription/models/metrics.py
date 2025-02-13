from collections.abc import Callable
from typing import Literal

import torch
import torchmetrics
from constants import FRAMES_PER_SECOND, START_NOTE, VELOCITY_SCALE
from data_utils.midi_utils import Target2MIDI
from mir_eval.transcription_velocity import precision_recall_f1_overlap

"""TO:DO consolidate both metrics into one inherited base class"""


class FramewiseMetrics(torchmetrics.Metric):
    """
    Compute Framewise Metrics for Transcription.

    Adapted from https://github.com/bytedance/piano_transcription/blob/master/pytorch/evaluate.py

    Computes framewise MAE and F1 score for onset, offset, frame, and velocity.

    Args:
        n_classes (int): The number of classes in the output/target.
        p_threshold (float): The probability threshold for classification.
    """

    ONSET_MAE = "onset_mae"
    OFFSET_MAE = "offset_mae"
    VELOCITY_MAE = "velocity_mae"
    FRAME_F1 = "frame_f1"
    # ONSET_F1 = "onset_f1"
    # OFFSET_F1 = "offset_f1"
    FRAME_PRECISION = "frame_precision"
    FRAME_RECALL = "frame_recall"
    FRAME_ACCURACY = "frame_accuracy"
    METRICS = [ONSET_MAE, OFFSET_MAE, VELOCITY_MAE, FRAME_F1, FRAME_PRECISION, FRAME_RECALL, FRAME_ACCURACY]

    def __init__(self, n_classes: int, p_threshold: float = 0.3):
        super().__init__()
        self.p_threshold = p_threshold

        # Precision, Recall, F1, Accuracy
        self.F1 = torchmetrics.classification.MultilabelF1Score(n_classes, average="macro", multidim_average="global")
        self.precision = torchmetrics.classification.MultilabelPrecision(
            n_classes, average="macro", multidim_average="global"
        )
        self.recall = torchmetrics.classification.MultilabelRecall(
            n_classes, average="macro", multidim_average="global"
        )
        self.acc = torchmetrics.classification.MultilabelAccuracy(n_classes, average="macro", multidim_average="global")

        self.metrics_dict = FramewiseMetrics._empty_metric_dict()
        self.totals = FramewiseMetrics._empty_metric_dict()

    @classmethod
    def _empty_metric_dict(cls) -> dict[str, float]:
        return {m: torch.tensor(0, dtype=torch.float) for m in FramewiseMetrics.METRICS}

    @staticmethod
    def concat_dict(x: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if len(x) == 0:
            return dict()
        return {k: torch.cat([d[k] for d in x]) for k in x[0]}

    @staticmethod
    def apply_to_tensor_dict(x: dict[str, torch.Tensor], func: Callable) -> dict[str, torch.Tensor]:
        return {k: func(v) for k, v in x.items()}

    @staticmethod
    def mask_mae(
        preds: torch.Tensor, target, mask: torch.Tensor | None = None, reduction: Literal["mean", "sum"] = "sum"
    ) -> torch.Tensor:
        if mask is not None:
            target *= mask
            preds *= mask

        mae_tensor = torch.abs(target - preds)

        if reduction == "sum":
            return torch.sum(mae_tensor)

        elif reduction == "mean":
            if mask is None:
                return torch.mean(mae_tensor)
            else:
                return torch.sum(mae_tensor) / torch.clip(torch.sum(mask), 1e-8, torch.inf)
        else:
            raise Exception(f"Reduction {reduction} not supported")

    @staticmethod
    def mask_output(preds: torch.Tensor, target) -> torch.Tensor:
        """Mask indicates only evaluate where either prediction or ground truth exists"""
        return (torch.sign(target + preds - 0.01) + 1) / 2

    def _infer_cls(self, preds: torch.Tensor) -> torch.Tensor:
        return ((torch.sign(preds - self.p_threshold) + 1) // 2).int()

    def infer_cls(self, preds: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(preds, torch.Tensor):
            return self._infer_cls(preds)
        else:
            return FramewiseMetrics.apply_to_tensor_dict(preds, self._infer_cls)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # onset MAE
        mask = FramewiseMetrics.mask_output(preds["reg_onset_output"], target["reg_onset_roll"])
        onset_mae = FramewiseMetrics.mask_mae(preds["reg_onset_output"], target["reg_onset_roll"], mask=mask)
        self.metrics_dict[FramewiseMetrics.ONSET_MAE] += onset_mae
        self.totals[FramewiseMetrics.ONSET_MAE] += torch.sum(mask)

        # offset MAE
        mask = FramewiseMetrics.mask_output(preds["reg_offset_output"], target["reg_offset_roll"])
        offset_mae = FramewiseMetrics.mask_mae(preds["reg_offset_output"], target["reg_offset_roll"], mask=mask)
        self.metrics_dict[FramewiseMetrics.OFFSET_MAE] += offset_mae
        self.totals[FramewiseMetrics.OFFSET_MAE] += torch.sum(mask)

        # velocity MAE
        mask = target["onset_roll"]
        velocity_mae = FramewiseMetrics.mask_mae(
            preds["velocity_output"], target["velocity_roll"] / VELOCITY_SCALE, mask=mask
        )
        self.metrics_dict[FramewiseMetrics.VELOCITY_MAE] += velocity_mae
        self.totals[FramewiseMetrics.VELOCITY_MAE] += torch.sum(mask)

        # Convert preds to class labels
        preds = self.infer_cls(preds)

        if self.F1.device != preds["frame_output"].device:
            self.F1 = self.F1.to(preds["frame_output"].device)
            self.acc = self.acc.to(preds["frame_output"].device)
            self.precision = self.precision.to(preds["frame_output"].device)
            self.recall = self.recall.to(preds["frame_output"].device)

        # frame F1, precision, recall
        self.F1.update(preds["frame_output"].transpose(1, 2), target["frame_roll"].transpose(1, 2))
        self.precision.update(preds["frame_output"].transpose(1, 2), target["frame_roll"].transpose(1, 2))
        self.recall.update(preds["frame_output"].transpose(1, 2), target["frame_roll"].transpose(1, 2))
        self.acc.update(preds["frame_output"].transpose(1, 2), target["frame_roll"].transpose(1, 2))
        # onset F1
        # self.F1.update(preds["reg_onset_output"].transpose(1, 2), target["reg_onset_roll"].transpose(1, 2))

        # offset F1
        # self.F1.update(preds["reg_offset_output"].transpose(1, 2), target["reg_offset_roll"].transpose(1, 2))

    def compute(self) -> dict[str, float]:
        # clip totals to avoid division by zero
        self.totals = FramewiseMetrics.apply_to_tensor_dict(self.totals, lambda x: torch.clip(x, 1e-8, torch.inf))

        result = {k: (self.metrics_dict[k] / self.totals[k]) for k in FramewiseMetrics.METRICS}

        result[FramewiseMetrics.FRAME_F1] = self.F1.compute()
        result[FramewiseMetrics.FRAME_PRECISION] = self.precision.compute()
        result[FramewiseMetrics.FRAME_RECALL] = self.recall.compute()
        result[FramewiseMetrics.FRAME_ACCURACY] = self.acc.compute()

        self.metrics_dict = FramewiseMetrics._empty_metric_dict()
        self.totals = FramewiseMetrics._empty_metric_dict()

        return result


class NoteLevelMetrics(torchmetrics.Metric):
    NOTE_F1 = "note_f1"
    NOTE_PRECISION = "note_precision"
    NOTE_RECALL = "note_recall"

    METRICS = [NOTE_F1, NOTE_PRECISION, NOTE_RECALL]

    def __init__(
        self,
        n_classes: int,
        frames_per_second: int = FRAMES_PER_SECOND,
        p_threshold: float = 0.3,
        onset_tolerance: float = 0.05,
        offset_min_tolerance: float = 0.05,
        offset_ratio: float = 0.2,  # None | 0.2
    ):
        super().__init__()
        self.p_threshold = p_threshold
        self.n_classes = n_classes

        # Use p_threshold for all output thresholds, potentially add multiple thresholds if necessary
        self.midi_processor = Target2MIDI(
            frames_per_second,
            n_classes,
            START_NOTE,
            velocity_scale=VELOCITY_SCALE,
            onset_threshold=p_threshold,
            offset_threshold=p_threshold,
            frame_threshold=p_threshold,
            pedal_offset_threshold=p_threshold,
        )

        # TO:DO create torchmetrics wrapper for mir_eval.transcription.f_measure
        self.F1 = precision_recall_f1_overlap
        self.preds = list()
        self.targets = list()

        self.onset_tolerance = onset_tolerance
        self.offset_ratio = offset_ratio
        self.offset_min_tolerance = offset_min_tolerance

    @staticmethod
    def note_to_freq(notes: torch.Tensor):
        return 2 ** ((notes - 39) / 12) * 440

    @staticmethod
    def apply_to_tensor_dict(x: dict[str, torch.Tensor], func: Callable) -> dict[str, torch.Tensor]:
        return {k: func(v) for k, v in x.items()}

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # remove any element of target for pedals and rename keys to match with preds
        """TODO: consolidate pred/target keys to remove need for conversion during postprocessing"""
        target = {k.replace("roll", "output"): v for k, v in target.items() if "pedal" not in k.lower()}

        # flatten batch dimension
        preds = NoteLevelMetrics.apply_to_tensor_dict(preds, lambda x: x.view(-1, self.n_classes))
        target = NoteLevelMetrics.apply_to_tensor_dict(target, lambda x: x.view(-1, self.n_classes))

        self.preds.append(torch.tensor(self.midi_processor.process(preds)[0], device="cpu"))
        self.targets.append(torch.tensor(self.midi_processor.process(target)[0], device="cpu"))

    def compute(self):
        self.preds = torch.cat(self.preds, dim=0).cpu().numpy()
        self.targets = torch.cat(self.targets, dim=0).cpu().numpy()

        # # Detect piano notes from output_dict
        est_on_offs = self.preds[:, 0:2]
        est_midi_notes = self.preds[:, 2]
        est_vels = self.preds[:, 3] * self.midi_processor.velocity_scale

        ref_on_off_pairs = self.targets[:, 0:2]
        ref_midi_notes = self.targets[:, 2]
        ref_vels = self.targets[:, 3] * self.midi_processor.velocity_scale

        self.preds = list()
        self.targets = list()

        # Calculate note metrics

        (note_precision, note_recall, note_f1, _) = self.F1(
            ref_intervals=ref_on_off_pairs,
            ref_pitches=NoteLevelMetrics.note_to_freq(ref_midi_notes),
            ref_velocities=ref_vels,
            est_intervals=est_on_offs,
            est_pitches=NoteLevelMetrics.note_to_freq(est_midi_notes),
            est_velocities=est_vels,
            onset_tolerance=self.onset_tolerance,
            offset_ratio=self.offset_ratio,
            offset_min_tolerance=self.offset_min_tolerance,
        )

        return {
            NoteLevelMetrics.NOTE_F1: note_f1,
            NoteLevelMetrics.NOTE_PRECISION: note_precision,
            NoteLevelMetrics.NOTE_RECALL: note_recall,
        }
