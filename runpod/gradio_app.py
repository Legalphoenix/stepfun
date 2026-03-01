#!/usr/bin/env python3
"""Browser chat UI served from the RunPod container."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import gradio as gr

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from stepaudior1vllm import StepAudioR1  # noqa: E402

OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://127.0.0.1:9999/v1/chat/completions")
MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "Step-Audio-R1.1")
UI_PORT = int(os.getenv("UI_PORT", "7860"))

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful audio assistant. If audio is provided, reason from the audio and answer clearly."
)

CLIENT = StepAudioR1(api_url=OPENAI_API_URL, model_name=MODEL_NAME)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_response(text: str) -> str:
    cleaned = THINK_RE.sub("", text or "")
    return cleaned.strip() or "(No output)"


def _content_as_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("value") or "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("value") or ""))
        return " ".join([p for p in parts if p]).strip()
    return str(content or "")


def _build_messages(
    user_message: str,
    history: list[dict],
    audio_path: str | None,
    system_prompt: str,
) -> list[dict]:
    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    for item in history:
        role = item.get("role")
        text = _content_as_text(item.get("content")).strip()
        if not text:
            continue
        if role == "user":
            messages.append({"role": "human", "content": [{"type": "text", "text": text}]})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": text})

    content: list[dict] = []
    if user_message.strip():
        content.append({"type": "text", "text": user_message.strip()})
    if audio_path:
        content.append({"type": "audio", "audio": audio_path})
    if not content:
        raise ValueError("Please enter text or upload audio.")

    messages.append({"role": "human", "content": content})
    messages.append({"role": "assistant", "content": "<think>\n", "eot": False})
    return messages


def chat_fn(
    message: str,
    history: list[dict],
    audio_file: str | None,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        messages = _build_messages(message, history, audio_file, system_prompt)
        full_text = ""
        for _, text, _ in CLIENT.stream(
            messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            repetition_penalty=1.0,
            stop_token_ids=[151665],
        ):
            if text:
                full_text += text
        return _clean_response(full_text)
    except Exception as exc:  # noqa: BLE001
        return f"Model not ready or request failed: {exc}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Step-Audio-R1.1 Chat") as app:
        gr.Markdown(
            """
            # Step-Audio-R1.1 Chat
            Upload audio (optional) and ask questions in text.
            """
        )
        gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[
                gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio (optional)",
                ),
                gr.Textbox(
                    value=DEFAULT_SYSTEM_PROMPT,
                    lines=2,
                    label="System Prompt",
                ),
                gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature"),
                gr.Slider(256, 32768, value=4096, step=256, label="Max Tokens"),
            ],
        )
    return app


if __name__ == "__main__":
    demo = build_app()
    demo.queue(max_size=64, api_open=False).launch(
        server_name="0.0.0.0",
        server_port=UI_PORT,
        show_error=True,
    )
