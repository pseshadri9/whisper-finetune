import torch
import torch.nn.functional as F


def bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be
    deactivated when calculation BCE.
    """
    eps = 1e-7
    output = torch.clamp(output, eps, 1.0 - eps)
    matrix = -target * torch.log(output) - (1.0 - target) * torch.log(1.0 - output)
    return torch.sum(matrix * mask) / torch.sum(mask)


class RegressOnsetOffsetFrameVelocityLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = bce

    def forward(self, output_dict, target_dict):
        """High-resolution piano note regression loss, including onset regression,
        offset regression, velocity regression and frame-wise classification losses.
        """
        onset_loss = self._loss(
            output_dict["reg_onset_output"], target_dict["reg_onset_roll"], target_dict["mask_roll"]
        )
        offset_loss = self._loss(
            output_dict["reg_offset_output"], target_dict["reg_offset_roll"], target_dict["mask_roll"]
        )
        frame_loss = self._loss(output_dict["frame_output"], target_dict["frame_roll"], target_dict["mask_roll"])
        velocity_loss = self._loss(
            output_dict["velocity_output"], target_dict["velocity_roll"] / 128, target_dict["onset_roll"]
        )
        total_loss = onset_loss + offset_loss + frame_loss + velocity_loss
        return total_loss


class RegressPedalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = F.binary_cross_entropy

    def forward(self, output_dict, target_dict):
        """High-resolution piano pedal regression loss, including pedal onset
        regression, pedal offset regression and pedal frame-wise classification losses.
        """
        onset_pedal_loss = self._loss(
            output_dict["reg_pedal_onset_output"], target_dict["reg_pedal_onset_roll"][:, :, None]
        )
        offset_pedal_loss = self._loss(
            output_dict["reg_pedal_offset_output"], target_dict["reg_pedal_offset_roll"][:, :, None]
        )
        frame_pedal_loss = self._loss(output_dict["pedal_frame_output"], target_dict["pedal_frame_roll"][:, :, None])
        total_loss = onset_pedal_loss + offset_pedal_loss + frame_pedal_loss
        return total_loss
