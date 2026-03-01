#!/usr/bin/env python3
"""Stop a RunPod Pod by name (defaults to step-audio-r1-1)."""

from __future__ import annotations

import argparse
import os

import requests

API_BASE = "https://rest.runpod.io/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stop a RunPod Pod by name.")
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"), help="RunPod API key.")
    parser.add_argument("--name", default="step-audio-r1-1", help="Pod name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing RUNPOD_API_KEY. Set it and re-run.")

    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    pods_resp = requests.get(f"{API_BASE}/pods", headers=headers, timeout=60)
    if pods_resp.status_code >= 400:
        raise RuntimeError(f"GET /pods failed ({pods_resp.status_code}): {pods_resp.text}")

    pods = [pod for pod in pods_resp.json() if pod.get("name") == args.name]
    if not pods:
        print(f"No Pod found with name '{args.name}'.")
        return

    pods.sort(key=lambda p: p.get("createdAt", ""), reverse=True)
    pod = pods[0]
    pod_id = pod["id"]
    stop_resp = requests.post(f"{API_BASE}/pods/{pod_id}/stop", headers=headers, timeout=60)
    if stop_resp.status_code >= 400:
        raise RuntimeError(f"POST /pods/{pod_id}/stop failed ({stop_resp.status_code}): {stop_resp.text}")

    print(f"Stopped pod {pod_id} ({args.name}).")


if __name__ == "__main__":
    main()
