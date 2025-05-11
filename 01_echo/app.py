import numpy as np
from dotenv import load_dotenv
from fastrtc import ReplyOnPause, Stream, get_hf_turn_credentials
from numpy.typing import NDArray

load_dotenv()


def detection(audio: tuple[int, NDArray[np.int16]]):
    """When the VAD detects the turn is over, pass this audio to the handler.
    Then we do what we want with it and generate new audio.
    """
    yield audio


stream = Stream(
    handler=ReplyOnPause(detection),
    modality="audio",
    mode="send-receive",
    concurrency_limit=5,
    time_limit=None,
)


if __name__ == "__main__":
    import os

    if (mode := os.getenv("MODE", "UI")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(port=7860)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    