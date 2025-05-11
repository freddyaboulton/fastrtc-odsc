import os

import numpy as np
from dotenv import load_dotenv
from fastrtc import (
    ReplyOnPause,
    Stream,
    get_current_context,
    get_stt_model,
    get_tts_model,
)
from huggingface_hub import InferenceClient
from numpy.typing import NDArray

load_dotenv()

tts_model = get_tts_model()
client = InferenceClient(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    provider="auto",
    token=os.getenv("HF_TOKEN"),
)
stt_model = get_stt_model()

conversations: dict[str, list[dict[str, str]]] = {}


def response(user_audio: tuple[int, NDArray[np.int16]]):
    context = get_current_context()
    if context.webrtc_id not in conversations:
        conversations[context.webrtc_id] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can answer questions and help with tasks."
                    "Please return a short response (no more than one or two sentences) that will be converted to audio using a text-to-speech model. Avoid using markdown as the text-to-speech model does not know how to translate that to audio."
                ),
            }
        ]
    messages = conversations[context.webrtc_id]
    print(messages)

    transcription = stt_model.stt(user_audio)
    messages.append({"role": "user", "content": transcription})
    completion = client.chat.completions.create(  # type: ignore
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        messages=messages,  # type: ignore
        max_tokens=512,
    )
    response = completion.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": response})
    conversations[context.webrtc_id] = messages
    yield from tts_model.stream_tts_sync(response)


stream = Stream(
    ReplyOnPause(response),
    modality="audio",
    mode="send-receive",
)


if __name__ == "__main__":
    if (mode := os.getenv("MODE", "UI")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(port=7860)
    else:
        raise ValueError(f"Invalid mode: {mode}")
