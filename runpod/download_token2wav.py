#!/usr/bin/env python3
"""Download StepFun token2wav decoder assets into TOKEN2WAV_MODEL_DIR."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


REQUIRED_FILES = [
    "campplus.onnx",
    "flow.pt",
    "flow.yaml",
    "hift.pt",
    "speech_tokenizer_v2_25hz.onnx",
]


def main() -> None:
    model_id = os.getenv("TOKEN2WAV_MODEL_ID", "stepfun-ai/Step-Audio-2-mini")
    model_dir = Path(os.getenv("TOKEN2WAV_MODEL_DIR", "/workspace/Step-Audio-2-mini"))
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    token2wav_dir = model_dir / "token2wav"
    token2wav_dir.mkdir(parents=True, exist_ok=True)

    if all((token2wav_dir / name).exists() for name in REQUIRED_FILES):
        print(f"token2wav assets already present at {token2wav_dir}; skipping download.")
        return

    print(f"Downloading token2wav assets from {model_id} into {model_dir} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        token=hf_token,
        allow_patterns=["token2wav/*"],
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("token2wav asset download completed.")


if __name__ == "__main__":
    main()
