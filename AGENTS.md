# AGENTS.md

## Context
User wants **simple one-click voice conversation** with `stepfun-ai/Step-Audio-R1.1` on RunPod:
- speak/record audio input
- model responds with **playable audio output**
- no manual terminal workflow beyond possibly opening a launcher UI

Repository: `/Users/volumental/Desktop/stepfun`  
Remote: `https://github.com/Legalphoenix/stepfun`

## Critical User Requirement
Do **not** use generic fallback TTS for final solution.

The user explicitly rejected this. Current state includes a TTS fallback in:
- `runpod/launcher_ui.py` (uses `edge_tts`)

This must be removed unless user explicitly asks for fallback mode.

## What Has Been Done
- Imported upstream Step-Audio-R1 code.
- Added RunPod automation scripts (`runpod/deploy_pod.py`, `runpod/one_click.py`, etc.).
- Added local launcher UI (`runpod/launcher_ui.py`) and hosted UI scaffold (`runpod/gradio_app.py`).
- Added one-click scripts and README instructions.

Recent commits:
- `46b42b0` Add voice reply playback and clarify multimodal launcher UX (contains TTS fallback; likely needs revert/edit)
- `15a64e2`, `94c2043`, `ae64abc`, `98dd293`, `d18748f`

## Known Technical Facts (Important)
From `stepaudior1vllm.py`:
- Input audio is converted and sent as `input_audio` chunks.
- Response parser reads:
  - `tts_content.tts_text`
  - `tts_content.tts_audio` (currently parsed as token IDs via regex `<audio_x>`).

This indicates the backend can emit audio token content, but current code does **not** decode those tokens into waveform bytes for playback.

## Main Task for New Agent
Implement **true model-native voice out** (not synthetic TTS) end-to-end.

### Required steps
1. Verify official Step-Audio-R1.1 output format and decoding path from primary sources:
   - upstream repo
   - official model card / custom code
   - stepfun custom vLLM path if needed
2. Determine how `tts_content.tts_audio` should be converted into playable audio.
3. Implement decoding pipeline in this repo.
4. Update launcher + hosted UI to:
   - capture mic/upload audio
   - send to model
   - play back decoded model audio response
5. Remove TTS fallback code and dependencies unless user re-requests it.
6. Keep interface minimal and model-specific.

## UX Requirements (Non-negotiable)
- One-click entry for non-technical user:
  - `runpod/launcher.command` opens UI
  - user pastes RunPod API key in UI
  - click Start/Reuse
  - talk immediately
- Avoid exposing irrelevant controls.
- Only expose knobs if confirmed applicable to this model path.

## Validation Requirements
Before final response:
1. Run local UI smoke test (`launcher_ui.py` starts, returns HTTP 200).
2. Run real request against active pod and confirm:
   - non-empty audio output file from model-native path
   - audio playable in UI
3. Document exactly what was tested and results.

## If Blocked
If official open-source artifacts do not include required audio decoder:
1. State that clearly with concrete evidence and file references.
2. Provide closest compliant alternative and tradeoffs.
3. Ask user whether to proceed with that alternative.

## Security
RunPod API key was shared in chat. Recommend user rotate it after setup changes.
