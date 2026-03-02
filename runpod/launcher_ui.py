#!/usr/bin/env python3
"""Local launcher UI: start/reuse RunPod and embed hosted voice chat."""

from __future__ import annotations

import argparse
import html
import json
from typing import Any

import gradio as gr
import requests

import deploy_pod
import one_click


def _proxy_urls(pod_id: str, api_port: int = 9999, ui_port: int = 7860) -> tuple[str, str]:
    return (
        f"https://{pod_id}-{api_port}.proxy.runpod.net/v1/chat/completions",
        f"https://{pod_id}-{ui_port}.proxy.runpod.net",
    )


def _hosted_embed(ui_url: str) -> str:
    if not ui_url:
        return "<div style='padding:12px;border:1px solid #ddd;border-radius:12px;'>Start or check a pod to load hosted voice chat here.</div>"
    escaped = html.escape(ui_url, quote=True)
    return (
        "<div style='margin-top:8px'>"
        f"<iframe src='{escaped}' style='width:100%;height:820px;border:1px solid #ddd;border-radius:12px;' allow='microphone; autoplay'></iframe>"
        "</div>"
    )


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
        "next_step": "Speak in the embedded hosted UI below.",
    }
    return json.dumps(summary, indent=2)


def start_pod(
    runpod_api_key: str,
    hf_token: str,
    pod_name: str,
    wait_for_model: bool,
) -> tuple[str, str, str, str]:
    if not runpod_api_key.strip():
        return "Missing RunPod API key.", "", "", _hosted_embed("")

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

    return _pod_summary(pod, api_url, ui_url, model_ready), pod_id, ui_url, _hosted_embed(ui_url)


def check_status(runpod_api_key: str, pod_name: str) -> tuple[str, str, str, str]:
    if not runpod_api_key.strip():
        return "Missing RunPod API key.", "", "", _hosted_embed("")

    pod = one_click.find_latest_named_pod(runpod_api_key.strip(), pod_name.strip())
    if pod is None:
        return f"No pod found named '{pod_name}'.", "", "", _hosted_embed("")

    pod = one_click.api_request(runpod_api_key.strip(), "GET", f"/pods/{pod['id']}")
    api_url, ui_url = _proxy_urls(pod["id"])
    model_ready = False
    try:
        resp = requests.get(api_url.replace("/v1/chat/completions", "/v1/models"), timeout=20)
        model_ready = resp.status_code == 200
    except requests.RequestException:
        model_ready = False
    return _pod_summary(pod, api_url, ui_url, model_ready), pod["id"], ui_url, _hosted_embed(ui_url)


def stop_pod(runpod_api_key: str, pod_name: str) -> tuple[str, str, str, str]:
    if not runpod_api_key.strip():
        return "Missing RunPod API key.", "", "", _hosted_embed("")

    pod = one_click.find_latest_named_pod(runpod_api_key.strip(), pod_name.strip())
    if pod is None:
        return f"No pod found named '{pod_name}'.", "", "", _hosted_embed("")

    one_click.api_request(runpod_api_key.strip(), "POST", f"/pods/{pod['id']}/stop")
    return f"Stopped pod {pod['id']} ({pod_name}).", "", "", _hosted_embed("")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Step-Audio-R1.1 One-Click Launcher") as app:
        gr.Markdown(
            """
            # Step-Audio-R1.1 One-Click Launcher
            1) Paste RunPod API key
            2) Click Start / Reuse Pod
            3) Speak directly in the embedded hosted voice UI
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

        gr.Markdown("## Hosted Voice Chat")
        hosted_ui_embed = gr.HTML(value=_hosted_embed(""))

        start_btn.click(
            fn=start_pod,
            inputs=[runpod_key, hf_token, pod_name, wait_model],
            outputs=[status_json, pod_id_box, hosted_ui_link, hosted_ui_embed],
        )
        status_btn.click(
            fn=check_status,
            inputs=[runpod_key, pod_name],
            outputs=[status_json, pod_id_box, hosted_ui_link, hosted_ui_embed],
        )
        stop_btn.click(
            fn=stop_pod,
            inputs=[runpod_key, pod_name],
            outputs=[status_json, pod_id_box, hosted_ui_link, hosted_ui_embed],
        )

    return app


if __name__ == "__main__":
    build_app().queue(max_size=64, api_open=False).launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        show_error=True,
    )
