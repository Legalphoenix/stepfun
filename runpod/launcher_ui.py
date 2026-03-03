#!/usr/bin/env python3
"""Local launcher UI: start/reuse RunPod and embed hosted voice chat."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import gradio as gr
import requests

import deploy_pod
import one_click

POLL_SECONDS = 20
TIMEOUT_SECONDS = 7200
LOG_DIR = Path(__file__).resolve().parent / "launcher_logs"
MAX_LOG_LINES_IN_UI = 120
DEFAULT_NETWORK_VOLUME_ID = "4ngqvk0ocf"


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


def _build_deploy_args(
    api_key: str,
    hf_token: str,
    name: str,
    network_volume_id: str,
    data_center_ids: list[str] | None = None,
) -> argparse.Namespace:
    args = argparse.Namespace(
        api_key=api_key,
        hf_token=hf_token or None,
        name=name,
        network_volume_id=network_volume_id or None,
        data_center_ids=data_center_ids or None,
        api_port=9999,
        ui_port=7860,
        timeout_seconds=7200,
        poll_seconds=20,
    )
    return one_click.build_deploy_args(args)


def _resolve_volume_data_center(api_key: str, network_volume_id: str) -> list[str]:
    nv_id = network_volume_id.strip()
    if not nv_id:
        return []
    try:
        network_volume = one_click.api_request(api_key, "GET", f"/networkvolumes/{nv_id}")
    except Exception:  # noqa: BLE001
        return []
    data_center_id = str(network_volume.get("dataCenterId") or "").strip()
    return [data_center_id] if data_center_id else []


def _attached_network_volume_id(pod: dict[str, Any]) -> str:
    return str((pod.get("networkVolume") or {}).get("id") or pod.get("networkVolumeId") or "").strip()


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    return cleaned or "step-audio-r1-1"


def _resolve_log_path(session_log_path: str, pod_name: str) -> Path:
    if session_log_path.strip():
        path = Path(session_log_path.strip())
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return LOG_DIR / f"{stamp}-{_safe_name(pod_name)}.log"


def _fmt_field(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _append_log(log_path: Path, message: str, **fields: Any) -> None:
    details = ""
    if fields:
        parts = [f"{key}={_fmt_field(value)}" for key, value in fields.items()]
        details = " | " + ", ".join(parts)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc_ts()} | {message}{details}\n")


def _tail_log(log_path: Path, max_lines: int = MAX_LOG_LINES_IN_UI) -> str:
    if not log_path.exists():
        return ""
    with log_path.open("r", encoding="utf-8") as handle:
        return "".join(deque(handle, maxlen=max_lines))


def _ui_output(
    status_text: str,
    pod_id: str,
    hosted_ui_url: str,
    stage: str,
    log_path: Path,
) -> tuple[str, str, str, str, str, str, str, str]:
    log_path_str = str(log_path)
    return (
        status_text,
        pod_id,
        hosted_ui_url,
        _hosted_embed(hosted_ui_url if hosted_ui_url else ""),
        stage,
        _tail_log(log_path),
        log_path_str,
        log_path_str,
    )


def _probe_status(api_url: str, ui_url: str) -> tuple[int | None, int | None]:
    api_status: int | None = None
    ui_status: int | None = None
    try:
        api_resp = requests.get(api_url.replace("/v1/chat/completions", "/v1/models"), timeout=20)
        api_status = api_resp.status_code
    except requests.RequestException:
        api_status = None
    try:
        ui_resp = requests.get(ui_url, timeout=20, allow_redirects=True)
        ui_status = ui_resp.status_code
    except requests.RequestException:
        ui_status = None
    return api_status, ui_status


def _stage_from_probe(pod: dict[str, Any], api_status: int | None, ui_status: int | None) -> str:
    status = pod.get("desiredStatus")
    if status in {"EXITED", "TERMINATED"}:
        return f"Pod is {status}. Click Start / Reuse Pod."
    if api_status == 200 and ui_status == 200:
        return "Ready: API and hosted voice UI are live."
    if ui_status == 200 and api_status != 200:
        return "Hosted UI is live; model API is still warming up."
    if api_status == 200 and ui_status != 200:
        return "Model API is live; hosted UI is still starting."
    if not pod.get("publicIp") or not (pod.get("portMappings") or {}):
        return "Allocating pod networking and runtime ports."
    if api_status == 502 or ui_status == 502:
        return "Container is up; services are still booting (proxy 502)."
    if api_status == 404 or ui_status == 404:
        return "Proxy route not ready yet (404)."
    return "Pod is warming up; waiting for API/UI readiness."


def _estimate_progress(
    pod: dict[str, Any],
    api_status: int | None,
    ui_status: int | None,
    elapsed_seconds: int,
) -> tuple[int, int]:
    if api_status == 200 and ui_status == 200:
        return 100, 0
    if ui_status == 200 and api_status != 200:
        progress = min(98, 90 + int(elapsed_seconds / 120))
        return progress, max(1, 5 - int(elapsed_seconds / 120))
    if not pod.get("publicIp") or not (pod.get("portMappings") or {}):
        progress = max(3, min(25, int((elapsed_seconds / 240) * 25)))
        return progress, max(12, 22 - int(elapsed_seconds / 60))
    if api_status == 502 or ui_status == 502:
        progress = min(85, max(25, 25 + int(elapsed_seconds / 60) * 4))
        return progress, max(5, 20 - int(elapsed_seconds / 60))
    if api_status == 404 or ui_status == 404:
        progress = min(75, max(20, 20 + int(elapsed_seconds / 60) * 3))
        return progress, max(8, 18 - int(elapsed_seconds / 60))
    progress = min(88, max(15, 15 + int(elapsed_seconds / 60) * 3))
    return progress, max(6, 20 - int(elapsed_seconds / 60))


def _stage_with_estimate(base_stage: str, progress_percent: int, eta_minutes: int) -> str:
    if progress_percent >= 100:
        return f"{base_stage} (100%)"
    return f"{base_stage} (approx {progress_percent}% | ~{eta_minutes} min remaining)"


def _is_host_capacity_resume_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "not enough free gpus on the host machine" in text


def start_pod(
    runpod_api_key: str,
    hf_token: str,
    pod_name: str,
    network_volume_id: str,
    wait_for_model: bool,
    session_log_path: str,
) -> tuple[str, str, str, str, str, str, str, str]:
    _ = session_log_path  # Start action always creates a new session log file.
    clean_name = pod_name.strip() or "step-audio-r1-1"
    log_path = _resolve_log_path("", clean_name)
    _append_log(log_path, "Start/reuse requested", pod_name=clean_name, wait_for_model=wait_for_model)

    if not runpod_api_key.strip():
        stage = "Missing RunPod API key."
        _append_log(log_path, stage)
        yield _ui_output("Missing RunPod API key.", "", "", stage, log_path)
        return

    try:
        api_key = runpod_api_key.strip()
        desired_network_volume_id = network_volume_id.strip()
        resolved_data_center_ids = _resolve_volume_data_center(api_key, desired_network_volume_id)
        _append_log(
            log_path,
            "Resolved startup placement",
            network_volume_id=desired_network_volume_id or "none",
            resolved_data_centers=resolved_data_center_ids or ["auto"],
        )
        _append_log(log_path, "Looking up existing pod", pod_name=clean_name)
        yield _ui_output(
            json.dumps(
                {
                    "stage": "Looking up existing pod...",
                    "pod_name": clean_name,
                    "network_volume_id": desired_network_volume_id or None,
                },
                indent=2,
            ),
            "",
            "",
            "Checking for existing pod...",
            log_path,
        )

        pod = one_click.find_latest_named_pod(api_key, clean_name)
        if pod is not None:
            pod = one_click.api_request(api_key, "GET", f"/pods/{pod['id']}")
            expected_code_rev = deploy_pod.current_launcher_code_rev()
            current_code_rev = str((pod.get("env") or {}).get("LAUNCHER_CODE_REV") or "")
            if current_code_rev != expected_code_rev:
                stale_id = pod["id"]
                _append_log(
                    log_path,
                    "Existing pod uses stale launcher code; recreating pod",
                    stale_pod_id=stale_id,
                    stale_code_rev=current_code_rev or "none",
                    expected_code_rev=expected_code_rev,
                )
                one_click.api_request(api_key, "DELETE", f"/pods/{stale_id}")
                pod = None

        if pod is not None:
            current_network_volume_id = _attached_network_volume_id(pod)
            if desired_network_volume_id and current_network_volume_id != desired_network_volume_id:
                stale_id = pod["id"]
                _append_log(
                    log_path,
                    "Existing pod uses different storage; recreating to attach requested network volume",
                    stale_pod_id=stale_id,
                    stale_network_volume_id=current_network_volume_id or "none",
                    requested_network_volume_id=desired_network_volume_id,
                )
                one_click.api_request(api_key, "DELETE", f"/pods/{stale_id}")
                pod = None

        if pod is None:
            _append_log(log_path, "No existing pod found; creating a new pod")
            yield _ui_output(
                json.dumps(
                    {
                        "stage": "Creating pod...",
                        "pod_name": clean_name,
                        "network_volume_id": desired_network_volume_id or None,
                        "data_center_ids": resolved_data_center_ids or None,
                    },
                    indent=2,
                ),
                "",
                "",
                "Creating a new pod on RunPod...",
                log_path,
            )
            pod = deploy_pod.create_pod(
                api_key,
                _build_deploy_args(
                    api_key,
                    hf_token,
                    clean_name,
                    desired_network_volume_id,
                    resolved_data_center_ids,
                ),
            )
            _append_log(log_path, "Pod created", pod_id=pod.get("id"), desired_status=pod.get("desiredStatus"))
        else:
            _append_log(
                log_path,
                "Reusing existing pod",
                pod_id=pod.get("id"),
                desired_status=pod.get("desiredStatus"),
                attached_network_volume_id=_attached_network_volume_id(pod) or "none",
            )

        yield _ui_output(
            json.dumps(
                {
                    "stage": "Ensuring pod is running...",
                    "pod_id": pod.get("id"),
                    "status": pod.get("desiredStatus"),
                    "network_volume_id": _attached_network_volume_id(pod) or None,
                },
                indent=2,
            ),
            pod.get("id", ""),
            "",
            "Ensuring pod is running...",
            log_path,
        )

        try:
            pod = one_click.start_pod_if_needed(api_key, pod)
            _append_log(log_path, "Pod start check complete", pod_id=pod.get("id"), desired_status=pod.get("desiredStatus"))
        except RuntimeError as exc:
            if not _is_host_capacity_resume_error(exc):
                raise
            stale_id = pod["id"]
            _append_log(
                log_path,
                "Host GPU capacity unavailable; recreating pod",
                stale_pod_id=stale_id,
                error=str(exc),
            )
            one_click.api_request(api_key, "DELETE", f"/pods/{stale_id}")
            pod = deploy_pod.create_pod(
                api_key,
                _build_deploy_args(
                    api_key,
                    hf_token,
                    clean_name,
                    desired_network_volume_id,
                    resolved_data_center_ids,
                ),
            )
            pod = one_click.start_pod_if_needed(api_key, pod)
            _append_log(log_path, "Pod recreated after host capacity issue", pod_id=pod.get("id"))

        pod_id = pod["id"]
        api_url, ui_url = _proxy_urls(pod_id)

        start_time = time.time()
        deadline = start_time + TIMEOUT_SECONDS
        while time.time() < deadline:
            pod = one_click.api_request(api_key, "GET", f"/pods/{pod_id}")
            api_status, ui_status = _probe_status(api_url, ui_url)
            model_ready = api_status == 200
            ui_ready = ui_status == 200
            elapsed_seconds = int(max(0, time.time() - start_time))
            base_stage = _stage_from_probe(pod, api_status, ui_status)
            progress_percent, eta_minutes = _estimate_progress(pod, api_status, ui_status, elapsed_seconds)
            stage = _stage_with_estimate(base_stage, progress_percent, eta_minutes)
            summary = {
                "pod_id": pod.get("id"),
                "name": pod.get("name"),
                "status": pod.get("desiredStatus"),
                "last_status_change": pod.get("lastStatusChange"),
                "network_volume_id": _attached_network_volume_id(pod) or None,
                "api_url": api_url,
                "ui_url": ui_url,
                "model_ready": model_ready,
                "api_status": api_status,
                "ui_status": ui_status,
                "elapsed_seconds": elapsed_seconds,
                "progress_percent_estimate": progress_percent,
                "eta_minutes_estimate": eta_minutes,
                "stage": stage,
                "next_step": "Open hosted UI when ui_status is 200.",
            }
            _append_log(
                log_path,
                "Readiness poll",
                pod_id=pod_id,
                desired_status=pod.get("desiredStatus"),
                network_volume_id=_attached_network_volume_id(pod) or "none",
                api_status=api_status,
                ui_status=ui_status,
                elapsed_seconds=elapsed_seconds,
                progress_percent_estimate=progress_percent,
                stage=base_stage,
            )
            hosted_link = ui_url if ui_ready else ""
            yield _ui_output(json.dumps(summary, indent=2), pod_id, hosted_link, stage, log_path)

            if ui_ready and (model_ready or not wait_for_model):
                _append_log(log_path, "Pod ready for use", pod_id=pod_id, ui_ready=ui_ready, model_ready=model_ready)
                return
            time.sleep(POLL_SECONDS)

        timeout_msg = {
            "pod_id": pod_id,
            "name": clean_name,
            "status": "TIMEOUT",
            "network_volume_id": _attached_network_volume_id(pod) or None,
            "api_url": api_url,
            "ui_url": ui_url,
            "note": "Timed out waiting for readiness. Pod may still be warming up.",
            "log_file": str(log_path),
        }
        stage = "Timed out waiting for readiness. Check session log for startup details."
        _append_log(log_path, "Timed out waiting for readiness", pod_id=pod_id, timeout_seconds=TIMEOUT_SECONDS)
        yield _ui_output(json.dumps(timeout_msg, indent=2), pod_id, "", stage, log_path)
        return
    except Exception as exc:  # noqa: BLE001
        stage = f"Error: {exc}"
        _append_log(log_path, "Error during startup", error=str(exc))
        yield _ui_output(f"Error starting pod: {exc}", "", "", stage, log_path)
        return


def check_status(runpod_api_key: str, pod_name: str, session_log_path: str) -> tuple[str, str, str, str, str, str, str, str]:
    clean_name = pod_name.strip() or "step-audio-r1-1"
    log_path = _resolve_log_path(session_log_path, clean_name)

    if not runpod_api_key.strip():
        stage = "Missing RunPod API key."
        _append_log(log_path, stage)
        return _ui_output("Missing RunPod API key.", "", "", stage, log_path)

    pod = one_click.find_latest_named_pod(runpod_api_key.strip(), clean_name)
    if pod is None:
        stage = "No pod found."
        _append_log(log_path, "Pod lookup returned nothing", pod_name=clean_name)
        return _ui_output(f"No pod found named '{clean_name}'.", "", "", stage, log_path)

    pod = one_click.api_request(runpod_api_key.strip(), "GET", f"/pods/{pod['id']}")
    api_url, ui_url = _proxy_urls(pod["id"])
    api_status, ui_status = _probe_status(api_url, ui_url)
    model_ready = api_status == 200
    ui_ready = ui_status == 200
    progress_percent, eta_minutes = _estimate_progress(pod, api_status, ui_status, elapsed_seconds=0)
    stage = _stage_with_estimate(_stage_from_probe(pod, api_status, ui_status), progress_percent, eta_minutes)
    summary = {
        "pod_id": pod.get("id"),
        "name": pod.get("name"),
        "status": pod.get("desiredStatus"),
        "last_status_change": pod.get("lastStatusChange"),
        "network_volume_id": _attached_network_volume_id(pod) or None,
        "api_url": api_url,
        "ui_url": ui_url,
        "model_ready": model_ready,
        "api_status": api_status,
        "ui_status": ui_status,
        "progress_percent_estimate": progress_percent,
        "eta_minutes_estimate": eta_minutes,
        "stage": stage,
    }
    _append_log(
        log_path,
        "Manual status check",
        pod_id=pod.get("id"),
        desired_status=pod.get("desiredStatus"),
        network_volume_id=_attached_network_volume_id(pod) or "none",
        api_status=api_status,
        ui_status=ui_status,
        progress_percent_estimate=progress_percent,
    )
    return _ui_output(json.dumps(summary, indent=2), pod["id"], (ui_url if ui_ready else ""), stage, log_path)


def stop_pod(runpod_api_key: str, pod_name: str, session_log_path: str) -> tuple[str, str, str, str, str, str, str, str]:
    clean_name = pod_name.strip() or "step-audio-r1-1"
    log_path = _resolve_log_path(session_log_path, clean_name)

    if not runpod_api_key.strip():
        stage = "Missing RunPod API key."
        _append_log(log_path, stage)
        return _ui_output("Missing RunPod API key.", "", "", stage, log_path)

    pod = one_click.find_latest_named_pod(runpod_api_key.strip(), clean_name)
    if pod is None:
        stage = "No pod found."
        _append_log(log_path, "Stop requested but no pod found", pod_name=clean_name)
        return _ui_output(f"No pod found named '{clean_name}'.", "", "", stage, log_path)

    one_click.api_request(runpod_api_key.strip(), "POST", f"/pods/{pod['id']}/stop")
    stage = "Pod stopped."
    _append_log(log_path, "Pod stopped by user", pod_id=pod.get("id"), pod_name=clean_name)
    return _ui_output(f"Stopped pod {pod['id']} ({clean_name}).", "", "", stage, log_path)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Step-Audio-R1.1 One-Click Launcher") as app:
        gr.Markdown(
            """
            # Step-Audio-R1.1 One-Click Launcher
            1) Paste RunPod API key
            2) Confirm Network Volume ID
            3) Click Start / Reuse Pod
            4) Watch **Startup Stage** for live progress
            5) Use **Startup Log Stream** + **Session Log File** for full boot traces
            6) Speak directly in the hosted voice UI after UI/API are ready

            Note: with a network volume attached, model files are reused on future starts and cold starts are faster.
            """
        )

        with gr.Row():
            runpod_key = gr.Textbox(
                label="RunPod API Key",
                type="password",
                value=os.getenv("RUNPOD_API_KEY", ""),
                placeholder="rpa_...",
            )
            hf_token = gr.Textbox(label="HF Token (optional)", type="password")

        with gr.Row():
            pod_name = gr.Textbox(value="step-audio-r1-1", label="Pod Name")
            network_volume_id = gr.Textbox(
                value=DEFAULT_NETWORK_VOLUME_ID,
                label="Network Volume ID (optional)",
                placeholder="e.g. 4ngqvk0ocf",
            )
            wait_model = gr.Checkbox(value=False, label="Wait for full model readiness")

        with gr.Row():
            start_btn = gr.Button("Start / Reuse Pod", variant="primary")
            status_btn = gr.Button("Check Status")
            stop_btn = gr.Button("Stop Pod")

        status_json = gr.Code(label="Status", language="json")
        startup_stage = gr.Textbox(label="Startup Stage", interactive=False)
        startup_log = gr.Textbox(label="Startup Log Stream", interactive=False, lines=14)
        session_log_file = gr.Textbox(label="Session Log File", interactive=False)
        pod_id_box = gr.Textbox(label="Pod ID")
        hosted_ui_link = gr.Textbox(label="Hosted UI URL")
        session_log_state = gr.State(value="")

        gr.Markdown("## Hosted Voice Chat")
        hosted_ui_embed = gr.HTML(value=_hosted_embed(""))

        start_btn.click(
            fn=start_pod,
            inputs=[runpod_key, hf_token, pod_name, network_volume_id, wait_model, session_log_state],
            outputs=[
                status_json,
                pod_id_box,
                hosted_ui_link,
                hosted_ui_embed,
                startup_stage,
                startup_log,
                session_log_file,
                session_log_state,
            ],
        )
        status_btn.click(
            fn=check_status,
            inputs=[runpod_key, pod_name, session_log_state],
            outputs=[
                status_json,
                pod_id_box,
                hosted_ui_link,
                hosted_ui_embed,
                startup_stage,
                startup_log,
                session_log_file,
                session_log_state,
            ],
        )
        stop_btn.click(
            fn=stop_pod,
            inputs=[runpod_key, pod_name, session_log_state],
            outputs=[
                status_json,
                pod_id_box,
                hosted_ui_link,
                hosted_ui_embed,
                startup_stage,
                startup_log,
                session_log_file,
                session_log_state,
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
