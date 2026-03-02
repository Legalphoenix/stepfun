#!/usr/bin/env python3
"""StepFun-native audio token decoder helpers."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

MAX_AUDIO_TOKEN_ID_EXCLUSIVE = 6561


class NativeAudioDecoder:
    """Decode StepFun audio token IDs into playable WAV bytes."""

    def __init__(
        self,
        stepaudio2_repo_dir: str,
        token2wav_dir: str,
        default_prompt_wav: str,
        *,
        float16: bool = False,
    ) -> None:
        self.stepaudio2_repo_dir = Path(stepaudio2_repo_dir)
        self.token2wav_dir = Path(token2wav_dir)
        self.default_prompt_wav = Path(default_prompt_wav)
        self.float16 = bool(float16)
        self._decoder = None

    def _ensure_decoder(self):
        if self._decoder is not None:
            return self._decoder

        if not self.stepaudio2_repo_dir.exists():
            raise FileNotFoundError(
                f"Step-Audio2 repo not found at {self.stepaudio2_repo_dir}. "
                "Set STEP_AUDIO2_REPO_DIR or ensure runpod/start_vllm.sh cloned it."
            )
        if not self.token2wav_dir.exists():
            raise FileNotFoundError(
                f"token2wav model directory not found at {self.token2wav_dir}. "
                "Set TOKEN2WAV_DIR or ensure decoder weights were downloaded."
            )

        repo_dir_str = str(self.stepaudio2_repo_dir)
        if repo_dir_str not in sys.path:
            sys.path.insert(0, repo_dir_str)

        try:
            from token2wav import Token2wav  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import Step-Audio2 token2wav decoder. "
                "Ensure dependencies are installed: onnxruntime, s3tokenizer, hyperpyyaml, torchaudio."
            ) from exc

        logger.info("Loading token2wav decoder from %s", self.token2wav_dir)
        self._decoder = Token2wav(str(self.token2wav_dir), float16=self.float16)
        return self._decoder

    @staticmethod
    def normalize_tokens(tokens: Iterable[int]) -> list[int]:
        normalized: list[int] = []
        for token in tokens:
            value = int(token)
            if 0 <= value < MAX_AUDIO_TOKEN_ID_EXCLUSIVE:
                normalized.append(value)
        return normalized

    def decode(self, tokens: Iterable[int], prompt_wav: str | None = None) -> bytes:
        audio_tokens = self.normalize_tokens(tokens)
        if not audio_tokens:
            raise ValueError("No valid model-native audio tokens to decode.")

        prompt_path = Path(prompt_wav) if prompt_wav else self.default_prompt_wav
        if not prompt_path.exists():
            prompt_path = self.default_prompt_wav
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt WAV not found at {prompt_path}. "
                "Set TOKEN2WAV_PROMPT_WAV to a valid audio file."
            )

        decoder = self._ensure_decoder()
        return decoder(audio_tokens, str(prompt_path))


__all__ = ["NativeAudioDecoder", "MAX_AUDIO_TOKEN_ID_EXCLUSIVE"]
