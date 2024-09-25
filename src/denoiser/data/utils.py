import torchaudio


def load_audio(path: str, start: int = 0, end: int = None):
    num_frames = -1 if end is None else end - start
    audio, sr = torchaudio.load(path, frame_offset=start, num_frames=num_frames)
    return audio, sr
