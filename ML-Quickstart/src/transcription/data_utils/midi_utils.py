import collections
from collections.abc import Callable

import numpy as np
from constants import VELOCITY_SCALE
from data_utils import DataSegment, Processor
from mido import Message, MetaMessage, MidiFile, MidiTrack
from numba import njit
from torch import Tensor, from_numpy

"""
Adapted from https://github.com/bytedance/piano_transcription
"""


class MIDI2Target(Processor):
    def __init__(self, segment_seconds: float, frames_per_second: int, begin_note: int, classes_num: int):
        """Class for processing MIDI events to target.

        Args:
          segment_seconds: float
          frames_per_second: int
          begin_note: int, A0 MIDI note of a piano
          classes_num: int
        """
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.begin_note = begin_note
        self.classes_num = classes_num
        self.max_piano_note = self.classes_num - 1

    def process(  # noqa: PLR0912 , PLR0915 disable linter for now, refactor TBD
        self,
        start_time: float,
        midi_events_time: list[float],
        midi_events: list[str],
        extend_pedal: bool = True,
        note_shift: int = 0,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, float]], list[dict[str, float]]]:
        """Process MIDI events of an audio segment to target for training,
        includes:
        1. Parse MIDI events
        2. Prepare note targets
        3. Prepare pedal targets

        Args:
          start_time: float, start time of a segment
          midi_events_time: list of float, times of MIDI events of a recording,
            e.g. [0, 3.3, 5.1, ...]
          midi_events: list of str, MIDI events of a recording, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]
          extend_pedal, bool, True: Notes will be set to ON until pedal is
            released. False: Ignore pedal events.

        Returns:
          target_dict: {
            'onset_roll': (frames_num, classes_num),
            'offset_roll': (frames_num, classes_num),
            'reg_onset_roll': (frames_num, classes_num),
            'reg_offset_roll': (frames_num, classes_num),
            'frame_roll': (frames_num, classes_num),
            'velocity_roll': (frames_num, classes_num),
            'mask_roll':  (frames_num, classes_num),
            'pedal_onset_roll': (frames_num,),
            'pedal_offset_roll': (frames_num,),
            'reg_pedal_onset_roll': (frames_num,),
            'reg_pedal_offset_roll': (frames_num,),
            'pedal_frame_roll': (frames_num,)}

          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.64, 'offset_time': 697.00, 'velocity': 44},
            {'midi_note': 58, 'onset_time': 697.00, 'offset_time': 697.19, 'velocity': 50}
            ...]

          pedal_events: list of dict, e.g. [
            {'onset_time': 149.37, 'offset_time': 150.35},
            {'onset_time': 150.54, 'offset_time': 152.06},
            ...]
        """

        # ------ 1. Parse MIDI events ------
        # Search the begin index of a segment
        for bgn_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time:
                break
        """E.g., start_time: 709.0, bgn_idx: 18003, event_time: 709.0146"""

        # Search the end index of a segment
        for fin_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time + self.segment_seconds:
                break
        """E.g., start_time: 709.0, bgn_idx: 18196, event_time: 719.0115"""

        note_events = []
        """E.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]"""

        pedal_events = []
        """E.g. [
            {'onset_time': 696.46875, 'offset_time': 696.62604},
            {'onset_time': 696.8063, 'offset_time': 698.50836},
            ...]"""

        buffer_dict = {}  # Used to store onset of notes to be paired with offsets
        pedal_dict = {}  # Used to store onset of pedal to be paired with offset of pedal

        # Backtrack bgn_idx to earlier indexes: ex_bgn_idx, which is used for
        # searching cross segment pedal and note events. E.g.: bgn_idx: 1149,
        # ex_bgn_idx: 981
        _delta = int((fin_idx - bgn_idx) * 1.0)
        ex_bgn_idx = max(bgn_idx - _delta, 0)

        for i in range(ex_bgn_idx, fin_idx):
            # Parse MIDI messiage
            attribute_list = midi_events[i].split(" ")

            # Note
            if attribute_list[0] in ["note_on", "note_off"]:
                """E.g. attribute_list: ['note_on', 'channel=0', 'note=41', 'velocity=0', 'time=10']"""

                midi_note = int(attribute_list[2].split("=")[1])
                velocity = int(attribute_list[3].split("=")[1])

                # Onset
                if attribute_list[0] == "note_on" and velocity > 0:
                    buffer_dict[midi_note] = {"onset_time": midi_events_time[i], "velocity": velocity}

                # Offset
                elif midi_note in buffer_dict.keys():
                    note_events.append(
                        {
                            "midi_note": midi_note,
                            "onset_time": buffer_dict[midi_note]["onset_time"],
                            "offset_time": midi_events_time[i],
                            "velocity": buffer_dict[midi_note]["velocity"],
                        }
                    )
                    del buffer_dict[midi_note]

            # Pedal
            elif attribute_list[0] == "control_change" and attribute_list[2] == "control=64":
                """control=64 corresponds to pedal MIDI event. E.g.
                attribute_list: ['control_change', 'channel=0', 'control=64', 'value=45', 'time=43']"""

                ped_value = int(attribute_list[3].split("=")[1])
                if ped_value >= 64:
                    if "onset_time" not in pedal_dict:
                        pedal_dict["onset_time"] = midi_events_time[i]
                elif "onset_time" in pedal_dict:
                    pedal_events.append({"onset_time": pedal_dict["onset_time"], "offset_time": midi_events_time[i]})
                    pedal_dict = {}

        # Add unpaired onsets to events
        for midi_note in buffer_dict.keys():
            note_events.append(
                {
                    "midi_note": midi_note,
                    "onset_time": buffer_dict[midi_note]["onset_time"],
                    "offset_time": start_time + self.segment_seconds,
                    "velocity": buffer_dict[midi_note]["velocity"],
                }
            )

        # Add unpaired pedal onsets to data
        if "onset_time" in pedal_dict.keys():
            pedal_events.append(
                {"onset_time": pedal_dict["onset_time"], "offset_time": start_time + self.segment_seconds}
            )

        # Set notes to ON until pedal is released
        if extend_pedal:
            note_events = self.extend_pedal(note_events, pedal_events)

        # Prepare targets
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        reg_offset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))
        """mask_roll is used for masking out cross segment notes"""

        pedal_onset_roll = np.zeros(frames_num)
        pedal_offset_roll = np.zeros(frames_num)
        reg_pedal_onset_roll = np.ones(frames_num)
        reg_pedal_offset_roll = np.ones(frames_num)
        pedal_frame_roll = np.zeros(frames_num)

        # ------ 2. Get note targets ------
        # Process note events to target
        for note_event in note_events:
            """note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}"""

            piano_note = np.clip(note_event["midi_note"] - self.begin_note + note_shift, 0, self.max_piano_note)
            """There are 88 keys on a piano"""

            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((note_event["onset_time"] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event["offset_time"] - start_time) * self.frames_per_second))

                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1

                    offset_roll[fin_frame, piano_note] = 1
                    velocity_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = note_event["velocity"]

                    # Vector from the center of a frame to ground truth offset
                    reg_offset_roll[fin_frame, piano_note] = (note_event["offset_time"] - start_time) - (
                        fin_frame / self.frames_per_second
                    )

                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1

                        # Vector from the center of a frame to ground truth onset
                        reg_onset_roll[bgn_frame, piano_note] = (note_event["onset_time"] - start_time) - (
                            bgn_frame / self.frames_per_second
                        )

                    # Mask out segment notes
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            """Get regression targets"""
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])

        # Process unpaired onsets to target
        for midi_note in buffer_dict.keys():
            piano_note = np.clip(midi_note - self.begin_note + note_shift, 0, self.max_piano_note)
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((buffer_dict[midi_note]["onset_time"] - start_time) * self.frames_per_second))
                mask_roll[bgn_frame:, piano_note] = 0

        # ------ 3. Get pedal targets ------
        # Process pedal events to target
        for pedal_event in pedal_events:
            bgn_frame = int(round((pedal_event["onset_time"] - start_time) * self.frames_per_second))
            fin_frame = int(round((pedal_event["offset_time"] - start_time) * self.frames_per_second))

            if fin_frame >= 0:
                pedal_frame_roll[max(bgn_frame, 0) : fin_frame + 1] = 1

                pedal_offset_roll[fin_frame] = 1
                reg_pedal_offset_roll[fin_frame] = (pedal_event["offset_time"] - start_time) - (
                    fin_frame / self.frames_per_second
                )

                if bgn_frame >= 0:
                    pedal_onset_roll[bgn_frame] = 1
                    reg_pedal_onset_roll[bgn_frame] = (pedal_event["onset_time"] - start_time) - (
                        bgn_frame / self.frames_per_second
                    )

        # Get regression pedal targets
        reg_pedal_onset_roll = self.get_regression(reg_pedal_onset_roll)
        reg_pedal_offset_roll = self.get_regression(reg_pedal_offset_roll)

        target_dict = {
            "onset_roll": onset_roll,
            "offset_roll": offset_roll,
            "reg_onset_roll": reg_onset_roll,
            "reg_offset_roll": reg_offset_roll,
            "frame_roll": frame_roll,
            "velocity_roll": velocity_roll,
            "mask_roll": mask_roll,
            "reg_pedal_onset_roll": reg_pedal_onset_roll,
            "pedal_onset_roll": pedal_onset_roll,
            "pedal_offset_roll": pedal_offset_roll,
            "reg_pedal_offset_roll": reg_pedal_offset_roll,
            "pedal_frame_roll": pedal_frame_roll,
        }

        return target_dict, note_events, pedal_events

    def extend_pedal(self, note_events: list[dict], pedal_events: list[dict]) -> list[dict[str, float]]:
        """Update the offset of all notes until pedal is released.

        Args:
          note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
          pedal_events: list of dict, e.g., [
            {'onset_time': 696.46875, 'offset_time': 696.62604},
            {'onset_time': 696.8063, 'offset_time': 698.50836},
            ...]

        Returns:
          ex_note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        """
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)
        ex_note_events = []

        idx = 0  # Index of note events
        while pedal_events:  # Go through all pedal events
            pedal_event = pedal_events.popleft()
            buffer_dict = {}  # keys: midi notes, value for each key: event index

            while note_events:
                note_event = note_events.popleft()

                # If a note offset is between the onset and offset of a pedal,
                # Then set the note offset to when the pedal is released.
                if pedal_event["onset_time"] < note_event["offset_time"] < pedal_event["offset_time"]:
                    midi_note = note_event["midi_note"]

                    if midi_note in buffer_dict.keys():
                        """Multiple same note inside a pedal"""
                        _idx = buffer_dict[midi_note]
                        del buffer_dict[midi_note]
                        ex_note_events[_idx]["offset_time"] = note_event["onset_time"]

                    # Set note offset to pedal offset
                    note_event["offset_time"] = pedal_event["offset_time"]
                    buffer_dict[midi_note] = idx

                ex_note_events.append(note_event)
                idx += 1

                # Break loop and pop next pedal
                if note_event["offset_time"] > pedal_event["offset_time"]:
                    break

        while note_events:
            """Append left notes"""
            ex_note_events.append(note_events.popleft())

        return ex_note_events

    def get_regression(self, input) -> np.ndarray:
        """Get regression target. See Fig. 2 of [1] for an example.
        [1] Q. Kong, et al., High-resolution Piano Transcription with Pedals by
        Regressing Onsets and Offsets Times, 2020.

        input:
          input: (frames_num,)

        Returns: (frames_num,), e.g., [0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, ...]
        """
        step = 1.0 / self.frames_per_second
        output = np.ones_like(input)

        locts = np.where(input < 0.5)[0]
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0.0, 0.05) * 20
        output = 1.0 - output

        return output


class Target2MIDI(Processor):
    def __init__(
        self,
        frames_per_second: int,
        classes_num: int,
        begin_note: int,
        velocity_scale: int = VELOCITY_SCALE,
        onset_threshold: float = 0.3,
        offset_threshold: float = 0.3,
        frame_threshold: float = 0.3,
        pedal_offset_threshold: float = 0.3,
    ):
        """Postprocess the output probabilities of a transription model to MIDI
        events.

        Args:
          frames_per_second: int
          classes_num: int
          onset_threshold: float
          offset_threshold: float
          frame_threshold: float
          pedal_offset_threshold: float
        """
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.pedal_offset_threshold = pedal_offset_threshold
        self.begin_note = begin_note
        self.velocity_scale = velocity_scale

    def output_dict_to_midi_events(self, output_dict):
        """Main function. Post process model outputs to MIDI events.

        Args:
          output_dict: {
            'reg_onset_output': (segment_frames, classes_num),
            'reg_offset_output': (segment_frames, classes_num),
            'frame_output': (segment_frames, classes_num),
            'velocity_output': (segment_frames, classes_num),
            'reg_pedal_onset_output': (segment_frames, 1),
            'reg_pedal_offset_output': (segment_frames, 1),
            'pedal_frame_output': (segment_frames, 1)}

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83},
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]

          est_pedal_events: list of dict, e.g. [
            {'onset_time': 0.17, 'offset_time': 0.96},
            {'osnet_time': 1.17, 'offset_time': 2.65}]
        """

        # Post process piano note outputs to piano note and pedal events information
        (est_on_off_note_vels, est_pedal_on_offs) = self.process(output_dict)
        """est_on_off_note_vels: (events_num, 4), the four columns are: [onset_time, offset_time, piano_note, velocity],
        est_pedal_on_offs: (pedal_events_num, 2), the two columns are: [onset_time, offset_time]"""

        # Reformat notes to MIDI events
        est_note_events = self.detected_notes_to_events(est_on_off_note_vels)

        if est_pedal_on_offs is None:
            est_pedal_events = None
        else:
            est_pedal_events = self.detected_pedals_to_events(est_pedal_on_offs)

        return est_note_events, est_pedal_events

    def process(self, output_dict):  # noqa: PLR0912 , PLR0915 disable linter for now, refactor TBD
        """Postprocess the output probabilities of a transription model to MIDI
        events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num),
            'reg_offset_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'velocity_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (events_num, 4), the 4 columns are onset_time,
            offset_time, piano_note and velocity. E.g. [
             [39.74, 39.87, 27, 0.65],
             [11.98, 12.11, 33, 0.69],
             ...]

          est_pedal_on_offs: (pedal_events_num, 2), the 2 columns are onset_time
            and offset_time. E.g. [
             [0.17, 0.96],
             [1.17, 2.65],
             ...]
        """

        # ------ 1. Process regression outputs to binarized outputs ------
        # For example, onset or offset of [0., 0., 0.15, 0.30, 0.40, 0.35, 0.20, 0.05, 0., 0.]
        # will be processed to [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]

        # Calculate binarized onset output from regression output
        (onset_output, onset_shift_output) = self.get_binarized_output_from_regression(
            reg_output=output_dict["reg_onset_output"],
            threshold=self.onset_threshold,
            neighbour=2,
            is_monotonic_neighbour=Target2MIDI.is_monotonic_neighbour,
        )

        output_dict["onset_output"] = onset_output  # Values are 0 or 1
        output_dict["onset_shift_output"] = onset_shift_output

        # Calculate binarized offset output from regression output
        (offset_output, offset_shift_output) = self.get_binarized_output_from_regression(
            reg_output=output_dict["reg_offset_output"],
            threshold=self.offset_threshold,
            neighbour=4,
            is_monotonic_neighbour=Target2MIDI.is_monotonic_neighbour,
        )

        output_dict["offset_output"] = offset_output  # Values are 0 or 1
        output_dict["offset_shift_output"] = offset_shift_output

        if "reg_pedal_onset_output" in output_dict.keys():
            """Pedal onsets are not used in inference. Instead, frame-wise pedal
            predictions are used to detect onsets. We empirically found this is
            more accurate to detect pedal onsets."""
            pass

        if "reg_pedal_offset_output" in output_dict.keys():
            # Calculate binarized pedal offset output from regression output
            (pedal_offset_output, pedal_offset_shift_output) = self.get_binarized_output_from_regression(
                reg_output=output_dict["reg_pedal_offset_output"],
                threshold=self.pedal_offset_threshold,
                neighbour=4,
                is_monotonic_neighbour=Target2MIDI.is_monotonic_neighbour,
            )

            output_dict["pedal_offset_output"] = pedal_offset_output  # Values are 0 or 1
            output_dict["pedal_offset_shift_output"] = pedal_offset_shift_output

        # ------ 2. Process matrices results to event results ------
        # Detect piano notes from output_dict
        est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)

        if "reg_pedal_onset_output" in output_dict.keys():
            # Detect piano pedals from output_dict
            est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict)

        else:
            est_pedal_on_offs = None

        return est_on_off_note_vels, est_pedal_on_offs

    @staticmethod
    @njit
    def get_binarized_output_from_regression(reg_output, threshold, neighbour, is_monotonic_neighbour: Callable = None):
        """Calculate binarized output and shifts of onsets or offsets from the
        regression results.

        Args:
          reg_output: (frames_num, classes_num)
          threshold: float
          neighbour: int
          is_monotonic_neighbour: Callable
            - for compliance with numba.njit, pass static method as argument

        Returns:
          binary_output: (frames_num, classes_num)
          shift_output: (frames_num, classes_num)
        """
        binary_output = np.zeros_like(reg_output)
        shift_output = np.zeros_like(reg_output)
        (frames_num, classes_num) = reg_output.shape

        for k in range(classes_num):
            x = reg_output[:, k]
            for n in range(neighbour, frames_num - neighbour):
                if x[n] > threshold and is_monotonic_neighbour(x, n, neighbour):
                    binary_output[n, k] = 1

                    """See Section III-D in [1] for deduction.
                    [1] Q. Kong, et al., High-resolution Piano Transcription
                    with Pedals by Regressing Onsets and Offsets Times, 2020."""
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    shift_output[n, k] = shift

        return binary_output, shift_output

    @staticmethod
    @njit
    def is_monotonic_neighbour(x: np.ndarray, n: int, neighbour: int) -> bool:
        """
        Detect if values are monotonic on both sides of x[n].

        Args:
            x: A tensor of shape (frames_num,).
            n: Index of the current position.
            neighbour: Number of elements to check on either side.

        Returns:
            monotonic: bool
        """
        """Detect if values are monotonic in both side of x[n].

        Args:
          x: (frames_num,)
          n: int
          neighbour: int

        Returns:
          monotonic: bool
        """
        monotonic = True
        for i in range(neighbour):
            if x[n - i] < x[n - i - 1]:
                monotonic = False
            if x[n + i] < x[n + i + 1]:
                monotonic = False

        return monotonic

    def output_dict_to_detected_notes(self, output_dict):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets,
          MIDI notes and velocities. E.g.,
            [[39.7375, 39.7500, 27., 0.6638],
             [11.9824, 12.5000, 33., 0.6892],
             ...]
        """
        est_tuples = []
        est_midi_notes = []
        classes_num = output_dict["frame_output"].shape[-1]

        for piano_note in range(classes_num):
            """Detect piano notes"""
            est_tuples_per_note = Target2MIDI.note_detection_with_onset_offset_regress(
                frame_output=output_dict["frame_output"][:, piano_note],
                onset_output=output_dict["onset_output"][:, piano_note],
                onset_shift_output=output_dict["onset_shift_output"][:, piano_note],
                offset_output=output_dict["offset_output"][:, piano_note],
                offset_shift_output=output_dict["offset_shift_output"][:, piano_note],
                velocity_output=output_dict["velocity_output"][:, piano_note],
                frame_threshold=self.frame_threshold,
            )

            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        est_tuples = np.array(est_tuples)  # (notes, 5)
        """(notes, 5), the five columns are onset, offset, onset_shift,
        offset_shift and normalized_velocity"""

        est_midi_notes = np.array(est_midi_notes)  # (notes,)

        if len(est_tuples) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        velocities = est_tuples[:, 4]

        est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
        """(notes, 3), the three columns are onset_times, offset_times and velocity."""

        est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

        return est_on_off_note_vels

    def output_dict_to_detected_pedals(self, output_dict):
        """Postprocess output_dict to piano pedals.

        Args:
          output_dict: dict, e.g. {
            'pedal_frame_output': (frames_num,),
            'pedal_offset_output': (frames_num,),
            'pedal_offset_shift_output': (frames_num,),
            ...}

        Returns:
          est_on_off: (notes, 2), the two columns are pedal onsets and pedal
            offsets. E.g.,
              [[0.1800, 0.9669],
               [1.1400, 2.6458],
               ...]
        """

        est_tuples = self.pedal_detection_with_onset_offset_regress(
            frame_output=output_dict["pedal_frame_output"][:, 0],
            offset_output=output_dict["pedal_offset_output"][:, 0],
            offset_shift_output=output_dict["pedal_offset_shift_output"][:, 0],
            frame_threshold=0.5,
        )

        est_tuples = np.array(est_tuples)
        """(notes, 2), the two columns are pedal onsets and pedal offsets"""

        if len(est_tuples) == 0:
            return np.array([])

        else:
            onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
            offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
            est_on_off = np.stack((onset_times, offset_times), axis=-1)
            est_on_off = est_on_off.astype(np.float32)
            return est_on_off

    def detected_notes_to_events(self, est_on_off_note_vels):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 3), the three columns are onset_times,
            offset_times and velocity. E.g.
            [[32.8376, 35.7700, 0.7932],
             [37.3712, 39.9300, 0.8058],
             ...]

        Returns:
          midi_events, list, e.g.,
            [{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
             {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
             ...]
        """
        midi_events = []
        for i in range(est_on_off_note_vels.shape[0]):
            midi_events.append(
                {
                    "onset_time": est_on_off_note_vels[i][0],
                    "offset_time": est_on_off_note_vels[i][1],
                    "midi_note": int(est_on_off_note_vels[i][2]),
                    "velocity": int(est_on_off_note_vels[i][3] * self.velocity_scale),
                }
            )

        return midi_events

    def detected_pedals_to_events(self, pedal_on_offs):
        """Reformat detected pedal onset and offsets to events.

        Args:
          pedal_on_offs: (notes, 2), the two columns are pedal onsets and pedal
          offsets. E.g.,
            [[0.1800, 0.9669],
             [1.1400, 2.6458],
             ...]

        Returns:
          pedal_events: list of dict, e.g.,
            [{'onset_time': 0.1800, 'offset_time': 0.9669},
             {'onset_time': 1.1400, 'offset_time': 2.6458},
             ...]
        """
        if pedal_on_offs is None:
            return None
        pedal_events = []
        for i in range(len(pedal_on_offs)):
            pedal_events.append({"onset_time": pedal_on_offs[i, 0], "offset_time": pedal_on_offs[i, 1]})

        return pedal_events

    @staticmethod
    @njit
    def note_detection_with_onset_offset_regress(
        frame_output,
        onset_output,
        onset_shift_output,
        offset_output,
        offset_shift_output,
        velocity_output,
        frame_threshold,
    ):
        """Process prediction matrices to note events information.
        First, detect onsets with onset outputs. Then, detect offsets
        with frame and offset outputs.

        Args:
        frame_output: (frames_num,)
        onset_output: (frames_num,)
        onset_shift_output: (frames_num,)
        offset_output: (frames_num,)
        offset_shift_output: (frames_num,)
        velocity_output: (frames_num,)
        frame_threshold: float

        Returns:
        output_tuples: list of [bgn, fin, onset_shift, offset_shift, normalized_velocity],
        e.g., [
            [1821, 1909, 0.47498, 0.3048533, 0.72119445],
            [1909, 1947, 0.30730522, -0.45764327, 0.64200014],
            ...]
        """
        # Used in place of None for static typing for numba compilation
        MAX_VALUE = np.finfo(np.float32).max

        # dummy element to enforce typing
        output_tuples = [np.zeros((1, 5), dtype=np.float32)]
        bgn = MAX_VALUE
        frame_disappear = MAX_VALUE
        offset_occur = MAX_VALUE

        """For Clarity: (x == MAX_VALUE) == (x is None)"""

        for i in range(onset_output.shape[0]):
            if onset_output[i] == 1:
                """Onset detected"""
                if bgn != MAX_VALUE:
                    """Consecutive onsets. E.g., pedal is not released, but two
                    consecutive notes being played."""
                    fin = np.float32(max(i - 1, 0))
                    output_tuples.append(
                        np.array(
                            [[bgn, fin, onset_shift_output[int(bgn)], 0, velocity_output[int(bgn)]]], dtype=np.float32
                        )
                    )
                    frame_disappear, offset_occur = MAX_VALUE, MAX_VALUE
                bgn = np.float32(i)

            if bgn != MAX_VALUE and i > bgn:
                """If onset found, then search offset"""
                if frame_output[i] <= frame_threshold and (frame_disappear == MAX_VALUE):
                    """Frame disappear detected"""
                    frame_disappear = np.float32(i)

                if offset_output[i] == 1 and (offset_occur == MAX_VALUE):
                    """Offset detected"""
                    offset_occur = np.float32(i)

                if frame_disappear != MAX_VALUE:
                    if (offset_occur != MAX_VALUE) and offset_occur - bgn > frame_disappear - offset_occur:
                        """bgn --------- offset_occur --- frame_disappear"""
                        fin = offset_occur
                    else:
                        """bgn --- offset_occur --------- frame_disappear"""
                        fin = frame_disappear
                    output_tuples.append(
                        np.array(
                            [
                                [
                                    bgn,
                                    fin,
                                    onset_shift_output[int(bgn)],
                                    offset_shift_output[int(fin)],
                                    velocity_output[int(bgn)],
                                ]
                            ],
                            dtype=np.float32,
                        )
                    )
                    bgn, frame_disappear, offset_occur = MAX_VALUE, MAX_VALUE, MAX_VALUE

                if bgn != MAX_VALUE and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                    """Offset not detected"""
                    fin = np.float32(i)
                    output_tuples.append(
                        np.array(
                            [
                                [
                                    bgn,
                                    fin,
                                    onset_shift_output[int(bgn)],
                                    offset_shift_output[int(fin)],
                                    velocity_output[int(bgn)],
                                ]
                            ],
                            dtype=np.float32,
                        )
                    )
                    bgn, frame_disappear, offset_occur = MAX_VALUE, MAX_VALUE, MAX_VALUE

        # if no detected onsets, return zeros list
        if len(output_tuples) == 1:
            return list(np.zeros_like(output_tuples[0]))
        output_tuples = output_tuples[1:]

        # allocate concatenated array for numba compilation
        output_np = np.zeros((len(output_tuples), output_tuples[0].shape[1]), dtype=np.float32)
        for x in range(len(output_tuples)):
            output_np[x, :] = output_tuples[x]

        # Sort pairs by onsets
        output_np = output_np[output_np[:, 0].argsort()]

        return list(output_np)

    @staticmethod
    def pedal_detection_with_onset_offset_regress(frame_output, offset_output, offset_shift_output, frame_threshold):
        """Process prediction array to pedal events information.

        Args:
        frame_output: (frames_num,)
        offset_output: (frames_num,)
        offset_shift_output: (frames_num,)
        frame_threshold: float

        Returns:
        output_tuples: list of [bgn, fin, onset_shift, offset_shift],
        e.g., [
            [1821, 1909, 0.4749851, 0.3048533],
            [1909, 1947, 0.30730522, -0.45764327],
            ...]
        """
        output_tuples = []
        bgn = None
        frame_disappear = None
        offset_occur = None

        for i in range(1, frame_output.shape[0]):
            if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
                """Pedal onset detected"""
                if bgn:
                    pass
                else:
                    bgn = i

            if bgn and i > bgn:
                """If onset found, then search offset"""
                if frame_output[i] <= frame_threshold and not frame_disappear:
                    """Frame disappear detected"""
                    frame_disappear = i

                if offset_output[i] == 1 and not offset_occur:
                    """Offset detected"""
                    offset_occur = i

                if offset_occur:
                    fin = offset_occur
                    output_tuples.append([bgn, fin, 0.0, offset_shift_output[fin]])
                    bgn, frame_disappear, offset_occur = None, None, None

                if frame_disappear and i - frame_disappear >= 10:
                    """offset not detected but frame disappear"""
                    fin = frame_disappear
                    output_tuples.append([bgn, fin, 0.0, offset_shift_output[fin]])
                    bgn, frame_disappear, offset_occur = None, None, None

        # Sort pairs by onsets
        output_tuples.sort(key=lambda pair: pair[0])

        return output_tuples

    @staticmethod
    ###### Google's onsets and frames post processing. Only used for comparison ######
    def onsets_frames_note_detection(frame_output, onset_output, offset_output, velocity_output, threshold):
        """Process pedal prediction matrices to note events information. onset_ouput
        is used to detect the presence of notes. frame_output is used to detect the
        offset of notes.

        Args:
        frame_output: (frames_num,)
        onset_output: (frames_num,)
        threshold: float

        Returns:
        bgn_fin_pairs: list of [bgn, fin, velocity]. E.g.
            [[1821, 1909, 0.47498, 0.72119445],
            [1909, 1947, 0.30730522, 0.64200014],
            ...]
        """
        output_tuples = []

        loct = None
        for i in range(onset_output.shape[0]):
            # Use onset_output is used to detect the presence of notes
            if onset_output[i] > threshold:
                if loct:
                    output_tuples.append([loct, i, velocity_output[loct]])
                loct = i
            if loct and i > loct:
                # Use frame_output is used to detect the offset of notes
                if frame_output[i] <= threshold:
                    output_tuples.append([loct, i, velocity_output[loct]])
                    loct = None

        output_tuples.sort(key=lambda pair: pair[0])

        return output_tuples

    @staticmethod
    def onsets_frames_pedal_detection(frame_output, offset_output, frame_threshold):
        """Process pedal prediction matrices to pedal events information.

        Args:
        frame_output: (frames_num,)
        offset_output: (frames_num,)
        offset_shift_output: (frames_num,)
        frame_threshold: float

        Returns:
        output_tuples: list of [bgn, fin],
        e.g., [
            [1821, 1909],
            [1909, 1947],
            ...]
        """
        output_tuples = []
        bgn = None
        frame_disappear = None
        offset_occur = None

        for i in range(1, frame_output.shape[0]):
            if frame_output[i] >= frame_threshold and frame_output[i] > frame_output[i - 1]:
                if bgn:
                    pass
                else:
                    bgn = i

            if bgn and i > bgn:
                """If onset found, then search offset"""
                if frame_output[i] <= frame_threshold and not frame_disappear:
                    """Frame disappear detected"""
                    frame_disappear = i

                if offset_output[i] == 1 and not offset_occur:
                    """Offset detected"""
                    offset_occur = i

                if offset_occur:
                    fin = offset_occur
                    output_tuples.append([bgn, fin])
                    bgn, frame_disappear, offset_occur = None, None, None

                if frame_disappear and i - frame_disappear >= 10:
                    """offset not detected but frame disappear"""
                    fin = frame_disappear
                    output_tuples.append([bgn, fin])
                    bgn, frame_disappear, offset_occur = None, None, None

        # Sort pairs by onsets
        output_tuples.sort(key=lambda pair: pair[0])

        return output_tuples

    @staticmethod
    def write_events_to_midi(start_time, note_events, pedal_events, midi_path):
        """Write out note events to MIDI file.

        Args:
        start_time: float
        note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        midi_path: str
        """

        # This configuration is the same as MIDIs in MAESTRO dataset
        ticks_per_beat = 384
        beats_per_second = 2
        ticks_per_second = ticks_per_beat * beats_per_second
        microseconds_per_beat = int(1e6 // beats_per_second)

        midi_file = MidiFile()
        midi_file.ticks_per_beat = ticks_per_beat

        # Track 0
        track0 = MidiTrack()
        track0.append(MetaMessage("set_tempo", tempo=microseconds_per_beat, time=0))
        track0.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        track0.append(MetaMessage("end_of_track", time=1))
        midi_file.tracks.append(track0)

        # Track 1
        track1 = MidiTrack()

        # Message rolls of MIDI
        message_roll = []

        for note_event in note_events:
            # Onset
            message_roll.append(
                {
                    "time": note_event["onset_time"],
                    "midi_note": note_event["midi_note"],
                    "velocity": note_event["velocity"],
                }
            )

            # Offset
            message_roll.append(
                {"time": note_event["offset_time"], "midi_note": note_event["midi_note"], "velocity": 0}
            )

        if pedal_events is not None:
            for pedal_event in pedal_events:
                message_roll.append({"time": pedal_event["onset_time"], "control_change": 64, "value": 127})
                message_roll.append({"time": pedal_event["offset_time"], "control_change": 64, "value": 0})

        # Sort MIDI messages by time
        message_roll.sort(key=lambda note_event: note_event["time"])

        previous_ticks = 0
        for message in message_roll:
            this_ticks = int((message["time"] - start_time) * ticks_per_second)
            if this_ticks >= 0:
                diff_ticks = this_ticks - previous_ticks
                previous_ticks = this_ticks
                if "midi_note" in message.keys():
                    track1.append(
                        Message("note_on", note=message["midi_note"], velocity=message["velocity"], time=diff_ticks)
                    )
                elif "control_change" in message.keys():
                    track1.append(
                        Message(
                            "control_change",
                            channel=0,
                            control=message["control_change"],
                            value=message["value"],
                            time=diff_ticks,
                        )
                    )
        track1.append(MetaMessage("end_of_track", time=1))
        midi_file.tracks.append(track1)

        midi_file.save(midi_path)


class MIDIFile(DataSegment):
    def __init__(self, filepath: str, processor: MIDI2Target) -> None:
        """
        Initializes an instance of the MIDIFile class.

        Args:
            filepath (str): The path to the MIDI file.
            start_time (float): The start time of the MIDI file.
            processor (MIDI2Target): The processor to be used for processing the MIDI file.

        Returns:
            None
        """
        self.filepath = filepath
        self.processor = processor

    def return_sample(self, start_time: float, end_time: float) -> dict[Tensor]:
        """
        Returns a sample of the MIDI file between the given start and end time.

        end_time of -1 returns till the end of the file

        Args:
            start_time (float): The start time of the sample.
            end_time (float): The end time of the sample.

        Returns:
            Tensor(s): Requisite data from the resultant segment.
        """
        midi_dict = self.read_file(self.filepath)

        midi_events_time, midi_events = (midi_dict["midi_event_time"], midi_dict["midi_event"])

        return_dict, _, _ = self.processor.process(start_time, midi_events_time, midi_events)
        start_frame = start_time * self.processor.frames_per_second
        end_frame = (end_time * self.processor.frames_per_second) + 1

        if (end_frame - start_frame >= return_dict["onset_roll"].shape[0]) or (end_time == -1):
            end_frame = return_dict["onset_roll"].shape[0]
            start_frame = 0

        # v[: round(end_frame - start_frame)]

        return {k: from_numpy(v) for k, v in return_dict.items()}

    @staticmethod
    def read_file(filepath: str) -> dict:
        """Parse MIDI file.

        Args:
        midi_path: str

        Returns:
        midi_dict: dict, e.g. {
            'midi_event': [
                'program_change channel=0 program=0 time=0',
                'control_change channel=0 control=64 value=127 time=0',
                'control_change channel=0 control=64 value=63 time=236',
                ...],
            'midi_event_time': [0., 0, 0.98307292, ...]}
        """

        midi_file = MidiFile(filepath)
        ticks_per_beat = midi_file.ticks_per_beat

        assert len(midi_file.tracks) == 2
        """The first track contains tempo, time signature. The second track
        contains piano events."""

        microseconds_per_beat = midi_file.tracks[0][0].tempo
        beats_per_second = 1e6 / microseconds_per_beat
        ticks_per_second = ticks_per_beat * beats_per_second

        message_list = []

        ticks = 0
        time_in_second = []

        for message in midi_file.tracks[1]:
            message_list.append(str(message))
            ticks += message.time
            time_in_second.append(ticks / ticks_per_second)

        midi_dict = {"midi_event": np.array(message_list), "midi_event_time": np.array(time_in_second)}

        return midi_dict


class CachedMIDIFile(DataSegment):
    NAMESPACE = ["reg_onset_roll", "reg_offset_roll", "frame_roll", "velocity_roll", ".mid"]
    """
    MIDI FILE ".mid" WILL BE USED TO RETRIEVE MASK_ROLL,
    FOR PROPER SAVING OF MASK_ROLL, MIDI2TARGET NEEDS TO BE REFACTORED, THIS IS A TEMPORARY FIX
    """

    def __init__(
        self,
        filepaths: list[str],
        frames_per_second: int,
        pre_processor: MIDI2Target,
        post_processor: Target2MIDI | None = None,
    ):
        self.filepaths = filepaths
        self.pre_processor = pre_processor  # In the future, pre-processor can be removed for processing mask_roll
        self.post_processor = post_processor
        self.frames_per_second = frames_per_second
        self.files = self.read_file(self.filepaths)

    def read_file(self, filepaths: list[str]) -> dict:
        file_map = dict()

        missing_files = []
        for name in CachedMIDIFile.NAMESPACE:
            found = False
            for f in filepaths:
                if name in f:
                    file_map[name] = f
                    found = True
                    break
            if not found:
                missing_files.append(name)

        assert len(missing_files) == 0, f"Missing files {missing_files}"

        for k, v in file_map.items():
            ### TEMPORARY HACK TO READ MASK ROLL FROM MIDI ###
            if k == ".mid":
                continue
            ##################################################
            else:
                file_map[k] = np.load(v, mmap_mode="r")

        ### TEMPORARY HACK TO READ MASK ROLL FROM MIDI ###
        file_map["mask_roll"] = MIDIFile(file_map[".mid"], self.pre_processor)
        del file_map[".mid"]
        ##################################################

        return file_map

    def return_sample(self, start_time: float, end_time: float) -> dict[Tensor]:
        start_frame = round(self.frames_per_second * start_time)
        end_frame = round(self.frames_per_second * end_time)

        return_dict = {k: from_numpy(v[start_frame:end_frame]) for k, v in self.files.items() if k != "mask_roll"}

        # Read mask_roll from midi
        return_dict["mask_roll"] = self.files["mask_roll"].return_sample(start_time, end_time)["mask_roll"]
        return return_dict

    @staticmethod
    def from_midi(midi_path: str, processor: MIDI2Target):
        raise NotImplementedError
