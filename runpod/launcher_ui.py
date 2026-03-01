#!/usr/bin/env python3
"""Local launcher UI: paste RunPod API key, click start, then chat."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import requests

import deploy_pod
import one_click

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from stepaudior1vllm import StepAudioR1  # noqa: E402

MODEL_NAME = "Step-Audio-R1.1"
_CLIENTS: dict[str, StepAudioR1] = {}


def _proxy_urls(pod_id: str, api_port: int = 9999, ui_port: int = 7860) -> tuple[str, str]:
    return (
        f"https://{pod_id}-{api_port}.proxy.runpod.net/v1/chat/completions",
        f"https://{pod_id}-{ui_port}.proxy.runpod.net",
    )


def _clean_text(text: str) -> str:
    return THINK_RE.sub("", text or "").strip() or "(No output)"


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


def _extract_message(message: str | dict) -> tuple[str, str | None]:
    if isinstance(message, str):
        return message.strip(), None

    text = str(message.get("text") or "").strip()
    files = message.get("files") or []
    audio_path: str | None = None
    for file_item in files:
        if isinstance(file_item, dict):
            path = file_item.get("path")
            if path:
                audio_path = str(path)
                break
        elif isinstance(file_item, str):
            audio_path = file_item
            break
    return text, audio_path


def _client_for_pod(pod_id: str) -> StepAudioR1:
    if pod_id not in _CLIENTS:
        api_url, _ = _proxy_urls(pod_id)
        _CLIENTS[pod_id] = StepAudioR1(api_url=api_url, model_name=MODEL_NAME)
    return _CLIENTS[pod_id]


def _build_deploy_args(api_key: str, hf_token: str, name: str) -> argparse.Namespace:
    args = argparse.Namespace(
        api_key=api_key,
        hf_token=hf_token or None,
        name=name,
        api_port=9999,
        ui_port=7860,
        timeout_seconds=7200,
        poll_seconds=20,
    )
    return one_click.build_deploy_args(args)


def _pod_summary(pod: dict[str, Any], api_url: str, ui_url: str, model_ready: bool) -> str:
    summary = {
        "pod_id": pod.get("id"),
        "name": pod.get("name"),
        "status": pod.get("desiredStatus"),
        "api_url": api_url,
        "ui_url": ui_url,
        "model_ready": model_ready,
        "next_step": "Open ui_url and chat.",
    }
    return json.dumps(summary, indent=2)


def start_pod(runpod_api_key: str, hf_token: str, pod_name: str, wait_for_model: bool) -> tuple[str, str, str]:
    if not runpod_api_key.strip():
        return "Missing RunPod API key.", "", ""

    pod = one_click.find_latest_named_pod(runpod_api_key.strip(), pod_name.strip())
    if pod is None:
        pod = deploy_pod.create_pod(runpod_api_key.strip(), _build_deploy_args(runpod_api_key, hf_token, pod_name))

    pod = one_click.start_pod_if_needed(runpod_api_key.strip(), pod)
    pod_id = pod["id"]
    api_url, ui_url = _proxy_urls(pod_id)

    model_ready = False
    try:
        _, _, model_ready = one_click.wait_for_ready(
            api_key=runpod_api_key.strip(),
            pod_id=pod_id,
            api_port=9999,
            ui_port=7860,
            wait_for_model=wait_for_model,
            timeout_seconds=7200,
            poll_seconds=20,
        )
    except TimeoutError:
        pass

    return _pod_summary(pod, api_url, ui_url, model_ready), pod_id, ui_url


def check_status(runpod_api_key: str, pod_name: str) -> tuple[str, str, str]:
    if not runpod_api_key.strip():
        return "Missing RunPod API key.", "", ""
    pod = one_click.find_latest_named_pod(runpod_api_key.strip(), pod_name.strip())
    if pod is None:
        return f"No pod found named '{pod_name}'.", "", ""
    pod = one_click.api_request(runpod_api_key.strip(), "GET", f"/pods/{pod['id']}")
    api_url, ui_url = _proxy_urls(pod["id"])
    model_ready = False
    try:
        resp = requests.get(api_url.replace("/v1/chat/completions", "/v1/models"), timeout=20)
        model_ready = resp.status_code == 200
    except requests.RequestException:
        model_ready = False
    return _pod_summary(pod, api_url, ui_url, model_ready), pod["id"], ui_url


def stop_pod(runpod_api_key: str, pod_name: str) -> str:
    if not runpod_api_key.strip():
        return "Missing RunPod API key."
    pod = one_click.find_latest_named_pod(runpod_api_key.strip(), pod_name.strip())
    if pod is None:
        return f"No pod found named '{pod_name}'."
    one_click.api_request(runpod_api_key.strip(), "POST", f"/pods/{pod['id']}/stop")
    return f"Stopped pod {pod['id']} ({pod_name})."


def chat_with_model(
    message: str | dict,
    history: list[dict],
    pod_id: str,
) -> str:
    if not pod_id.strip():
        return "Set Pod ID first by clicking Start/Reuse Pod."

    user_text, audio_path = _extract_message(message)
    content: list[dict[str, str]] = []
    if user_text:
        content.append({"type": "text", "text": user_text})
    if audio_path:
        content.append({"type": "audio", "audio": audio_path})
    if not content:
        return "Type a message or upload/record audio."

    model_messages: list[dict] = []
    for item in history:
        role = item.get("role")
        text = _content_as_text(item.get("content")).strip()
        if not text:
            continue
        if role == "user":
            model_messages.append({"role": "human", "content": [{"type": "text", "text": text}]})
        elif role == "assistant":
            model_messages.append({"role": "assistant", "content": text})

    model_messages.append({"role": "human", "content": content})
    model_messages.append({"role": "assistant", "content": "<think>\n", "eot": False})

    try:
        full_text = ""
        client = _client_for_pod(pod_id.strip())
        for _, text, _ in client.stream(
            model_messages,
            max_tokens=4096,
            temperature=0.7,
            repetition_penalty=1.0,
            stop_token_ids=[151665],
        ):
            if text:
                full_text += text
        return _clean_text(full_text)
    except Exception as exc:  # noqa: BLE001
        return f"Request failed: {exc}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Step-Audio-R1.1 One-Click Launcher") as app:
        gr.Markdown(
            """
            # Step-Audio-R1.1 One-Click Launcher
            1) Paste RunPod API key
            2) Click Start/Reuse Pod
            3) Chat directly in this page with text + audio
            """
        )

        with gr.Row():
            runpod_key = gr.Textbox(label="RunPod API Key", type="password", placeholder="rpa_...")
            hf_token = gr.Textbox(label="HF Token (optional)", type="password")

        with gr.Row():
            pod_name = gr.Textbox(value="step-audio-r1-1", label="Pod Name")
            wait_model = gr.Checkbox(value=False, label="Wait for full model readiness")

        with gr.Row():
            start_btn = gr.Button("Start / Reuse Pod", variant="primary")
            status_btn = gr.Button("Check Status")
            stop_btn = gr.Button("Stop Pod")

        status_json = gr.Code(label="Status", language="json")
        pod_id_box = gr.Textbox(label="Pod ID")
        hosted_ui_link = gr.Textbox(label="Hosted UI URL")

        start_btn.click(
            fn=start_pod,
            inputs=[runpod_key, hf_token, pod_name, wait_model],
            outputs=[status_json, pod_id_box, hosted_ui_link],
        )
        status_btn.click(
            fn=check_status,
            inputs=[runpod_key, pod_name],
            outputs=[status_json, pod_id_box, hosted_ui_link],
        )
        stop_btn.click(
            fn=stop_pod,
            inputs=[runpod_key, pod_name],
            outputs=[status_json],
        )

        gr.Markdown("## Chat (text + audio)")
        gr.ChatInterface(
            fn=chat_with_model,
            multimodal=True,
            additional_inputs=[
                pod_id_box,
            ],
        )

    return app


if __name__ == "__main__":
    build_app().queue(max_size=64, api_open=False).launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        show_error=True,
    )
