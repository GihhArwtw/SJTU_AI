from pathlib import Path


def parse_vad_label(line, frame_size: float = 0.032, frame_shift: float = 0.008):
    """Parse VAD information in each line, and convert it to frame-wise VAD label.

    Args:
        line (str): e.g. "0.2,3.11 3.48,10.51 10.52,11.02"
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
    Returns:
        frames (List[int]): frame-wise VAD label

    Examples:
        >>> label = parse_vad_label("0.3,0.5 0.7,0.9")
        [0, ..., 0, 1, ..., 1, 0, ..., 0, 1, ..., 1]
        >>> print(len(label))
        110

    NOTE: The output label length may vary according to the last timestamp in `line`,
    which may not correspond to the real duration of that sample.

    For example, if an audio sample contains 1-sec silence at the end, the resulting
    VAD label will be approximately 1-sec shorter than the sample duration.

    Thus, you need to pad zeros manually to the end of each label to match the number
    of frames in the feature. E.g.:
        >>> feature = extract_feature(audio)    # frames: 320
        >>> frames = feature.shape[1]           # here assumes the frame dimention is 1
        >>> label = parse_vad_label(vad_line)   # length: 210
        >>> import numpy as np
        >>> label_pad = np.pad(label, (0, np.maximum(frames - len(label), 0)))[:frames]
    """
    frame2time = lambda n: n * frame_shift + frame_size / 2
    frames = []
    frame_n = 0
    for time_pairs in line.split():
        start, end = map(float, time_pairs.split(","))
        assert end > start, (start, end)
        while frame2time(frame_n) < start:
            frames.append(0)
            frame_n += 1
        while frame2time(frame_n) <= end:
            frames.append(1)
            frame_n += 1
    return frames


def prediction_to_vad_label(
    prediction,
    frame_size: float = 0.032,
    frame_shift: float = 0.008,
    threshold: float = 0.5,
):
    """Convert model prediction to VAD labels.

    Args:
        prediction (List[float]): predicted speech activity of each **frame** in one sample
            e.g. [0.01, 0.03, 0.48, 0.66, 0.89, 0.87, ..., 0.72, 0.55, 0.20, 0.18, 0.07]
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
        threshold (float): prediction values that are higher than `threshold` are set to 1,
                            and those lower than or equal to `threshold` are set to 0
    Returns:
        vad_label (str): converted VAD label
            e.g. "0.31,2.56 2.6,3.89 4.62,7.99 8.85,11.06"

    NOTE: Each frame is converted to the timestamp according to its center time point.
    Thus the converted labels may not exactly coincide with the original VAD label, depending
    on the specified `frame_size` and `frame_shift`.
    See the following exmaple for more detailed explanation.

    Examples:
        >>> label = parse_vad_label("0.31,0.52 0.75,0.92")
        >>> prediction_to_vad_label(label)
        '0.31,0.53 0.75,0.92'
    """
    frame2time = lambda n: n * frame_shift + frame_size / 2
    speech_frames = []
    prev_state = False
    start, end = 0, 0
    end_prediction = len(prediction) - 1
    for i, pred in enumerate(prediction):
        state = pred > threshold
        if not prev_state and state:
            # 0 -> 1
            start = i
        elif not state and prev_state:
            # 1 -> 0
            end = i
            speech_frames.append(
                "{:.2f},{:.2f}".format(frame2time(start), frame2time(end))
            )
        elif i == end_prediction and state:
            # 1 -> 1 (end)
            end = i
            speech_frames.append(
                "{:.2f},{:.2f}".format(frame2time(start), frame2time(end))
            )
        prev_state = state
    return " ".join(speech_frames)


##############################################
# Examples of how to use the above functions #
##############################################
def read_label_from_file(
    path="data/dev_label.txt", frame_size: float = 0.032, frame_shift: float = 0.008
):
    """Read VAD information of all samples, and convert into
    frame-wise labels (not padded yet).

    Args:
        path (str): Path to the VAD label file.
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
    Returns:
        data (dict): Dictionary storing the frame-wise VAD
                    information of each sample.
            e.g. {
                "1031-133220-0062": [0, 0, 0, 0, ... ],
                "1031-133220-0091": [0, 0, 0, 0, ... ],
                ...
            }
    """
    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.strip().split(maxsplit=1)
            if len(sps) == 1:
                print(f'Error happened with path="{path}", id="{sps[0]}", value=""')
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = parse_vad_label(v, frame_size=frame_size, frame_shift=frame_shift)
    return data
