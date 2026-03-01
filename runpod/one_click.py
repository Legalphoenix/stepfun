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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click RunPod startup for Step-Audio-R1.1")
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"), help="RunPod API key.")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Optional Hugging Face token.")
    parser.add_argument("--name", default="step-audio-r1-1", help="Pod name to reuse/create.")
    parser.add_argument("--api-port", type=int, default=9999)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--poll-seconds", type=int, default=20)
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
    desired_status = pod.get("desiredStatus")
    if desired_status == "EXITED":
        api_request(api_key, "POST", f"/pods/{pod_id}/start")
    return api_request(api_key, "GET", f"/pods/{pod_id}")


def wait_for_api_ready(
    api_key: str,
    pod_id: str,
    api_port: int,
    timeout_seconds: int,
    poll_seconds: int,
) -> str:
    deadline = time.time() + timeout_seconds
    proxy_url = f"https://{pod_id}-{api_port}.proxy.runpod.net/v1/models"
    while time.time() < deadline:
        pod = api_request(api_key, "GET", f"/pods/{pod_id}")
        if pod.get("desiredStatus") in {"TERMINATED"}:
            raise RuntimeError(f"Pod {pod_id} terminated unexpectedly.")
        try:
            resp = requests.get(proxy_url, timeout=30)
            if resp.status_code == 200:
                return f"https://{pod_id}-{api_port}.proxy.runpod.net/v1/chat/completions"
        except requests.RequestException:
            pass
        time.sleep(poll_seconds)
    raise TimeoutError(
        "Timed out waiting for model readiness. Pod may still be downloading weights; re-run the same command."
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
        max_model_len=16384,
        max_num_seqs=32,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
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

    if args.no_wait:
        print(
            json.dumps(
                {
                    "pod_id": pod_id,
                    "name": pod.get("name"),
                    "status": pod.get("desiredStatus"),
                    "proxy_base_url": proxy_base,
                    "api_url": f"{proxy_base}/v1/chat/completions",
                },
                indent=2,
            )
        )
        return

    api_url = wait_for_api_ready(
        api_key=args.api_key,
        pod_id=pod_id,
        api_port=args.api_port,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    print(
        json.dumps(
            {
                "pod_id": pod_id,
                "name": args.name,
                "api_url": api_url,
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
