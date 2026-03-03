#!/usr/bin/env python3
"""Hosted voice chat UI served from the RunPod container."""

from __future__ import annotations

import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

import gradio as gr

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from runpod.native_audio import NativeAudioDecoder  # noqa: E402
from stepaudior1vllm import StepAudioR1  # noqa: E402

OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://127.0.0.1:9999/v1/chat/completions")
MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "Step-Audio-R1.1")
UI_PORT = int(os.getenv("UI_PORT", "7860"))
MAX_RESPONSE_TOKENS = int(os.getenv("MAX_RESPONSE_TOKENS", "768"))
CACHE_DIR = Path(os.getenv("GRADIO_TEMP_DIR", "/tmp/stepaudio-r1"))
STEP_AUDIO2_REPO_DIR = os.getenv("STEP_AUDIO2_REPO_DIR", "/workspace/Step-Audio2")
TOKEN2WAV_DIR = os.getenv("TOKEN2WAV_DIR", "/workspace/Step-Audio-2-mini/token2wav")
TOKEN2WAV_PROMPT_WAV = os.getenv(
    "TOKEN2WAV_PROMPT_WAV",
    f"{STEP_AUDIO2_REPO_DIR}/assets/default_female.wav",
)
TOKEN2WAV_FLOAT16 = os.getenv("TOKEN2WAV_FLOAT16", "0") in {"1", "true", "True"}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful audio assistant. Respond naturally and clearly to what the user says."
)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

CLIENT = StepAudioR1(api_url=OPENAI_API_URL, model_name=MODEL_NAME)
DECODER = NativeAudioDecoder(
    stepaudio2_repo_dir=STEP_AUDIO2_REPO_DIR,
    token2wav_dir=TOKEN2WAV_DIR,
    default_prompt_wav=TOKEN2WAV_PROMPT_WAV,
    float16=TOKEN2WAV_FLOAT16,
)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(CACHE_DIR)


def _clean_response(text: str) -> str:
    cleaned = THINK_RE.sub("", text or "")
    cleaned = cleaned.replace("<tts_start>", "").replace("<tts_end>", "")
    return cleaned.strip() or "(No output)"


def _save_tmp_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(dir=CACHE_DIR, delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        return f.name


def _base_model_history() -> list[dict]:
    return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]


def reset_chat() -> tuple[list[dict], list[dict], str, None, str, None]:
    return [], _base_model_history(), "", None, "", None


def submit_turn(
    chatbot: list[dict],
    model_history: list[dict],
    text: str,
    mic_audio: str | None,
) -> tuple[list[dict], list[dict], str, None, str, str | None]:
    chatbot = list(chatbot or [])
    model_history = list(model_history or _base_model_history())

    user_text = (text or "").strip()
    audio_path = mic_audio if mic_audio and Path(mic_audio).exists() else None

    if not user_text and not audio_path:
        return chatbot, model_history, "", None, "Please type or record audio.", None

    user_content: list[dict[str, str]] = []
    if audio_path:
        user_content.append({"type": "audio", "audio": audio_path})
        chatbot.append({"role": "user", "content": {"path": audio_path}})
    if user_text:
        user_content.append({"type": "text", "text": user_text})
        chatbot.append({"role": "user", "content": user_text})

    model_history.append({"role": "user", "content": user_content})

    try:
        request_messages = model_history + [
            {"role": "assistant", "content": "<tts_start>", "eot": False}
        ]

        response, reply_text_raw, reply_audio_tokens = CLIENT(
            request_messages,
            max_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.7,
            repetition_penalty=1.0,
            stop_token_ids=[151665],
        )

        reply_text = _clean_response(reply_text_raw)
        last_audio_path: str | None = None

        if reply_audio_tokens:
            prompt_wav = audio_path or TOKEN2WAV_PROMPT_WAV
            audio_bytes = DECODER.decode(reply_audio_tokens, prompt_wav=prompt_wav)
            last_audio_path = _save_tmp_audio(audio_bytes)
            if reply_text and reply_text != "(No output)":
                chatbot.append({"role": "assistant", "content": reply_text})
            chatbot.append({"role": "assistant", "content": {"path": last_audio_path}})

            tts_content = response.get("tts_content", {}) if isinstance(response, dict) else {}
            if isinstance(tts_content, dict) and tts_content.get("tts_audio"):
                model_history.append({"role": "assistant", "tts_content": tts_content})
            else:
                model_history.append({"role": "assistant", "content": reply_text})
        else:
            chatbot.append({"role": "assistant", "content": reply_text})
            model_history.append({"role": "assistant", "content": reply_text})

        return chatbot, model_history, "", None, "", last_audio_path
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        error = f"Request failed: {exc}"
        chatbot.append({"role": "assistant", "content": error})
        return chatbot, model_history, "", None, error, None


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Step-Audio-R1.1 Voice Chat") as app:
        gr.Markdown(
            """
            # Step-Audio-R1.1 Voice Chat
            Speak into the mic or upload audio, then click **Send**.
            This UI uses model-native audio output decoding (no synthetic TTS).
            """
        )

        chatbot = gr.Chatbot(label="Conversation", type="messages", min_height=520)
        model_history = gr.State(_base_model_history())

        with gr.Row():
            mic_audio = gr.Audio(
                label="Your Voice",
                type="filepath",
                sources=["microphone", "upload"],
                format="wav",
            )
            text = gr.Textbox(label="Optional Text", placeholder="Add text (optional)", lines=3)

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            stop_btn = gr.Button("Stop Reply")
            clear_btn = gr.Button("Clear")

        status = gr.Textbox(label="Status", interactive=False)
        assistant_audio = gr.Audio(
            label="Assistant Voice",
            type="filepath",
            autoplay=True,
            interactive=False,
        )

        submit_event = send_btn.click(
            fn=submit_turn,
            inputs=[chatbot, model_history, text, mic_audio],
            outputs=[chatbot, model_history, text, mic_audio, status, assistant_audio],
            concurrency_id="voice_chat",
            concurrency_limit=2,
        )
        text.submit(
            fn=submit_turn,
            inputs=[chatbot, model_history, text, mic_audio],
            outputs=[chatbot, model_history, text, mic_audio, status, assistant_audio],
            concurrency_id="voice_chat",
            concurrency_limit=2,
        )
        stop_btn.click(fn=None, cancels=[submit_event])
        clear_btn.click(
            fn=reset_chat,
            inputs=None,
            outputs=[chatbot, model_history, text, mic_audio, status, assistant_audio],
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.queue(max_size=64, api_open=False).launch(
        server_name="0.0.0.0",
        server_port=UI_PORT,
        show_error=True,
    )
