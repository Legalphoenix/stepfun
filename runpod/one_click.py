#!/usr/bin/env python3
"""One-click RunPod launcher for Step-Audio-R1.1.

Behavior:
1) Reuse a Pod with the target name if it already exists.
2) Otherwise create a new Pod.
3) Start the Pod if it is stopped.
4) Wait until the vLLM OpenAI endpoint is live via RunPod proxy URL.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import requests

import deploy_pod

API_BASE = "https://rest.runpod.io/v1"
REQUIRED_RUNTIME_ENV = {
    "MAX_MODEL_LEN": "4096",
    "MAX_NUM_SEQS": "4",
    "GPU_MEMORY_UTILIZATION": "0.97",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click RunPod startup for Step-Audio-R1.1")
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"), help="RunPod API key.")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Optional Hugging Face token.")
    parser.add_argument("--name", default="step-audio-r1-1", help="Pod name to reuse/create.")
    parser.add_argument("--api-port", type=int, default=9999)
    parser.add_argument("--ui-port", type=int, default=7860)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--wait-for-model", action="store_true", help="Wait for model API readiness, not only UI.")
    parser.add_argument("--no-wait", action="store_true", help="Return immediately after create/start.")
    return parser.parse_args()


def api_request(api_key: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.request(
        method=method,
        url=f"{API_BASE}{path}",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"RunPod API {method} {path} failed ({response.status_code}): {response.text}")
    if not response.content:
        return {}
    return response.json()


def list_pods(api_key: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{API_BASE}/pods",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"RunPod API GET /pods failed ({response.status_code}): {response.text}")
    return response.json()


def find_latest_named_pod(api_key: str, name: str) -> dict[str, Any] | None:
    pods = [pod for pod in list_pods(api_key) if pod.get("name") == name]
    if not pods:
        return None
    pods.sort(key=lambda p: p.get("createdAt", ""), reverse=True)
    return pods[0]


def start_pod_if_needed(api_key: str, pod: dict[str, Any]) -> dict[str, Any]:
    pod_id = pod["id"]
    env = dict(pod.get("env") or {})
    if any(str(env.get(key)) != value for key, value in REQUIRED_RUNTIME_ENV.items()):
        updated_env = dict(env)
        updated_env.update(REQUIRED_RUNTIME_ENV)
        api_request(api_key, "PATCH", f"/pods/{pod_id}", payload={"env": updated_env})
        pod = api_request(api_key, "GET", f"/pods/{pod_id}")

    desired_status = pod.get("desiredStatus")
    if desired_status == "EXITED":
        api_request(api_key, "POST", f"/pods/{pod_id}/start")
    return api_request(api_key, "GET", f"/pods/{pod_id}")


def wait_for_ready(
    api_key: str,
    pod_id: str,
    api_port: int,
    ui_port: int,
    wait_for_model: bool,
    timeout_seconds: int,
    poll_seconds: int,
) -> tuple[str, str, bool]:
    deadline = time.time() + timeout_seconds
    api_models_url = f"https://{pod_id}-{api_port}.proxy.runpod.net/v1/models"
    api_url = f"https://{pod_id}-{api_port}.proxy.runpod.net/v1/chat/completions"
    ui_url = f"https://{pod_id}-{ui_port}.proxy.runpod.net"
    ui_ready = False
    model_ready = False
    while time.time() < deadline:
        pod = api_request(api_key, "GET", f"/pods/{pod_id}")
        if pod.get("desiredStatus") in {"TERMINATED"}:
            raise RuntimeError(f"Pod {pod_id} terminated unexpectedly.")
        try:
            ui_resp = requests.get(ui_url, timeout=30, allow_redirects=True)
            if ui_resp.status_code in {200, 302, 307}:
                ui_ready = True
        except requests.RequestException:
            pass
        try:
            api_resp = requests.get(api_models_url, timeout=30)
            if api_resp.status_code == 200:
                model_ready = True
        except requests.RequestException:
            pass

        if ui_ready and (model_ready or not wait_for_model):
            return api_url, ui_url, model_ready
        time.sleep(poll_seconds)
    raise TimeoutError(
        "Timed out waiting for readiness. Pod may still be downloading weights; re-run the same command."
    )


def build_deploy_args(args: argparse.Namespace) -> argparse.Namespace:
    # Create an args object compatible with deploy_pod.create_pod()
    deploy_args = argparse.Namespace(
        api_key=args.api_key,
        name=args.name,
        cloud_type="SECURE",
        gpu_count=1,
        gpu_types=None,
        volume_gb=200,
        container_disk_gb=80,
        interruptible=False,
        model_id="stepfun-ai/Step-Audio-R1.1",
        model_dir="/workspace/Step-Audio-R1.1",
        served_model_name="Step-Audio-R1.1",
        api_port=args.api_port,
        ui_port=args.ui_port,
        max_model_len=4096,
        max_num_seqs=4,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.97,
        hf_token=args.hf_token,
        no_wait=True,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    return deploy_args


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing RUNPOD_API_KEY. Set it and re-run.")

    pod = find_latest_named_pod(args.api_key, args.name)
    if pod is None:
        pod = deploy_pod.create_pod(args.api_key, build_deploy_args(args))

    pod = start_pod_if_needed(args.api_key, pod)
    pod_id = pod["id"]
    proxy_base = f"https://{pod_id}-{args.api_port}.proxy.runpod.net"
    ui_base = f"https://{pod_id}-{args.ui_port}.proxy.runpod.net"

    if args.no_wait:
        print(
            json.dumps(
                {
                    "pod_id": pod_id,
                    "name": pod.get("name"),
                    "status": pod.get("desiredStatus"),
                    "proxy_base_url": proxy_base,
                    "api_url": f"{proxy_base}/v1/chat/completions",
                    "ui_url": ui_base,
                },
                indent=2,
            )
        )
        return

    api_url, ui_url, model_ready = wait_for_ready(
        api_key=args.api_key,
        pod_id=pod_id,
        api_port=args.api_port,
        ui_port=args.ui_port,
        wait_for_model=args.wait_for_model,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    print(
        json.dumps(
            {
                "pod_id": pod_id,
                "name": args.name,
                "api_url": api_url,
                "ui_url": ui_url,
                "model_ready": model_ready,
                "note": (
                    "Open ui_url in browser. If model_ready is false, the UI is up but model is still warming up."
                ),
                "test_command": (
                    "curl -s "
                    + api_url
                    + " -H 'Content-Type: application/json' "
                    + "-d '{\"model\":\"Step-Audio-R1.1\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one line.\"}]}'"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
