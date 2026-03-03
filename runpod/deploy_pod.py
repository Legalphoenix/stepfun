#!/usr/bin/env python3
"""Automate RunPod Pod deployment for Step-Audio-R1.1."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

API_BASE = "https://rest.runpod.io/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_RELATIVE_FILES = [
    "runpod/gradio_app.py",
    "runpod/native_audio.py",
    "runpod/chat_template.jinja",
    "runpod/start_vllm.sh",
    "stepaudior1vllm.py",
]

DEFAULT_GPU_TYPES = [
    "NVIDIA H200",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
]


def _overlay_sources() -> dict[str, str]:
    overlay: dict[str, str] = {}
    for rel_path in OVERLAY_RELATIVE_FILES:
        abs_path = REPO_ROOT / rel_path
        overlay[rel_path] = base64.b64encode(abs_path.read_bytes()).decode("ascii")
    return overlay


def current_launcher_code_rev() -> str:
    sha = hashlib.sha256()
    for rel_path in OVERLAY_RELATIVE_FILES:
        abs_path = REPO_ROOT / rel_path
        sha.update(rel_path.encode("utf-8"))
        sha.update(abs_path.read_bytes())
    return sha.hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and configure a Step-Audio Pod on RunPod.")
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"), help="RunPod API key.")
    parser.add_argument("--name", default="step-audio-r1-1", help="Pod name.")
    parser.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument(
        "--gpu-type",
        dest="gpu_types",
        action="append",
        help="Preferred GPU type (can be passed multiple times).",
    )
    parser.add_argument("--volume-gb", type=int, default=200)
    parser.add_argument("--container-disk-gb", type=int, default=80)
    parser.add_argument("--interruptible", action="store_true")
    parser.add_argument("--model-id", default="stepfun-ai/Step-Audio-R1.1")
    parser.add_argument("--model-dir", default="/workspace/Step-Audio-R1.1")
    parser.add_argument("--served-model-name", default="Step-Audio-R1.1")
    parser.add_argument("--api-port", type=int, default=9999)
    parser.add_argument("--ui-port", type=int, default=7860)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="Optional Hugging Face token.")
    parser.add_argument(
        "--network-volume-id",
        default=os.getenv("RUNPOD_NETWORK_VOLUME_ID"),
        help="Attach an existing RunPod network volume by ID.",
    )
    parser.add_argument(
        "--data-center-id",
        dest="data_center_ids",
        action="append",
        help="Preferred data center ID (can be passed multiple times).",
    )
    parser.add_argument("--no-wait", action="store_true", help="Return immediately after Pod creation.")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=10)
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


def build_start_command() -> str:
    overlay_blob_b64 = base64.b64encode(
        json.dumps(_overlay_sources(), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).decode("ascii")
    return (
        "set -euo pipefail; "
        "REPO_DIR=/workspace/stepfun; "
        "START_LOG=/workspace/runpod_start.log; "
        "mkdir -p /workspace; "
        "{ "
        "if [ -d \"$REPO_DIR/.git\" ]; then "
        "git -C \"$REPO_DIR\" pull --ff-only || echo 'WARN: git pull failed; continuing with existing checkout.'; "
        "elif [ -d \"$REPO_DIR\" ] && [ \"$(ls -A \"$REPO_DIR\" 2>/dev/null)\" ]; then "
        "echo 'Using existing non-git directory at /workspace/stepfun'; "
        "else git clone https://github.com/Legalphoenix/stepfun.git \"$REPO_DIR\"; fi; "
        "python3 - <<'PY'\n"
        "import base64\n"
        "import json\n"
        "from pathlib import Path\n"
        f"overlay = json.loads(base64.b64decode('{overlay_blob_b64}').decode('utf-8'))\n"
        "repo_dir = Path('/workspace/stepfun')\n"
        "for rel_path, file_b64 in overlay.items():\n"
        "    target = repo_dir / rel_path\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    target.write_bytes(base64.b64decode(file_b64))\n"
        "print(f'Applied launcher overlay to {len(overlay)} file(s).')\n"
        "PY\n"
        "chmod +x \"$REPO_DIR/runpod/start_vllm.sh\"; "
        "\"$REPO_DIR/runpod/start_vllm.sh\"; "
        "} >>\"$START_LOG\" 2>&1 || { "
        "code=$?; "
        "echo \"Startup failed with exit code $code. Inspect $START_LOG\" >>\"$START_LOG\"; "
        "tail -n 200 \"$START_LOG\"; "
        "sleep infinity; "
        "}"
    )


def create_pod(api_key: str, args: argparse.Namespace) -> dict[str, Any]:
    tensor_parallel_size = args.tensor_parallel_size or args.gpu_count
    gpu_types = args.gpu_types or DEFAULT_GPU_TYPES
    data_center_ids = [dc.strip() for dc in (args.data_center_ids or []) if dc and dc.strip()]
    if not data_center_ids:
        env_data_centers = os.getenv("RUNPOD_DATA_CENTER_IDS", "")
        if env_data_centers:
            data_center_ids = [dc.strip() for dc in env_data_centers.split(",") if dc.strip()]

    env = {
        "MODEL_ID": args.model_id,
        "MODEL_DIR": args.model_dir,
        "SERVED_MODEL_NAME": args.served_model_name,
        "API_HOST": "0.0.0.0",
        "API_PORT": str(args.api_port),
        "UI_PORT": str(args.ui_port),
        "MAX_MODEL_LEN": str(args.max_model_len),
        "MAX_NUM_SEQS": str(args.max_num_seqs),
        "TENSOR_PARALLEL_SIZE": str(tensor_parallel_size),
        "GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
        "LAUNCHER_CODE_REV": current_launcher_code_rev(),
    }
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token

    ports = [f"{args.api_port}/http", "22/tcp"]
    if args.ui_port != args.api_port:
        ports.insert(1, f"{args.ui_port}/http")

    payload = {
        "name": args.name,
        "cloudType": args.cloud_type,
        "computeType": "GPU",
        "gpuCount": args.gpu_count,
        "gpuTypeIds": gpu_types,
        "gpuTypePriority": "availability",
        "imageName": "stepfun2025/vllm:step-audio-2-v20250909",
        "containerDiskInGb": args.container_disk_gb,
        "ports": ports,
        "supportPublicIp": True,
        "interruptible": args.interruptible,
        "dockerStartCmd": ["bash", "-lc", build_start_command()],
        "env": env,
    }
    if data_center_ids:
        payload["dataCenterIds"] = data_center_ids
        payload["dataCenterPriority"] = "custom"
    if args.network_volume_id:
        payload["networkVolumeId"] = args.network_volume_id.strip()
        payload["volumeMountPath"] = "/workspace"
    else:
        payload["volumeInGb"] = args.volume_gb

    return api_request(api_key, "POST", "/pods", payload)


def wait_for_network_ready(
    api_key: str,
    pod_id: str,
    api_port: int,
    timeout_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        pod = api_request(api_key, "GET", f"/pods/{pod_id}")
        port_mappings = {str(k): v for k, v in (pod.get("portMappings") or {}).items()}
        desired_status = pod.get("desiredStatus")
        if desired_status in {"EXITED", "TERMINATED"}:
            raise RuntimeError(f"Pod moved to {desired_status}. Last status: {pod.get('lastStatusChange')}")
        if pod.get("publicIp") and port_mappings.get(str(api_port)):
            return pod
        time.sleep(poll_seconds)
    raise TimeoutError("Timed out waiting for Pod networking to become ready.")


def summarize(pod: dict[str, Any], api_port: int, ui_port: int) -> dict[str, Any]:
    port_mappings = {str(k): v for k, v in (pod.get("portMappings") or {}).items()}
    public_ip = pod.get("publicIp")
    mapped_api_port = port_mappings.get(str(api_port))
    mapped_ui_port = port_mappings.get(str(ui_port))
    mapped_ssh_port = port_mappings.get("22")
    pod_id = pod.get("id")

    api_url = None
    ui_url = None
    ssh_command = None
    if public_ip and mapped_api_port:
        api_url = f"http://{public_ip}:{mapped_api_port}/v1/chat/completions"
    if public_ip and mapped_ui_port:
        ui_url = f"http://{public_ip}:{mapped_ui_port}"
    if public_ip and mapped_ssh_port:
        ssh_command = f"ssh root@{public_ip} -p {mapped_ssh_port}"

    return {
        "pod_id": pod_id,
        "name": pod.get("name"),
        "desired_status": pod.get("desiredStatus"),
        "last_status_change": pod.get("lastStatusChange"),
        "gpu": pod.get("gpu", {}).get("displayName"),
        "gpu_count": pod.get("gpu", {}).get("count"),
        "public_ip": public_ip,
        "port_mappings": port_mappings,
        "api_url": api_url,
        "ui_url": ui_url,
        "api_proxy_url": (
            f"https://{pod_id}-{api_port}.proxy.runpod.net/v1/chat/completions" if pod_id else None
        ),
        "ui_proxy_url": f"https://{pod_id}-{ui_port}.proxy.runpod.net" if pod_id else None,
        "ssh_command": ssh_command,
    }


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing RunPod API key. Set RUNPOD_API_KEY or pass --api-key.")

    pod = create_pod(args.api_key, args)
    if args.no_wait:
        print(json.dumps(summarize(pod, args.api_port, args.ui_port), indent=2))
        return

    try:
        pod = wait_for_network_ready(
            api_key=args.api_key,
            pod_id=pod["id"],
            api_port=args.api_port,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        print(json.dumps(summarize(pod, args.api_port, args.ui_port), indent=2))
    except TimeoutError:
        current = api_request(args.api_key, "GET", f"/pods/{pod['id']}")
        summary = summarize(current, args.api_port, args.ui_port)
        summary["warning"] = (
            "Pod created but networking is not ready yet. Check again in RunPod UI or rerun "
            "this script with --no-wait to get IDs instantly."
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
