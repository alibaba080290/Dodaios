# French local speech on BC-250

This branch connects local Nemotron recognition and Siwis/Piper speech output
to the existing Claude subscription CLI for conversational responses.
The interface does not need a French translation.

## Integration

French now survives the reply pin, brain normalization, output resolver and
BCP-47 conversion. Piper/Sherpa 2 uses external eSpeak NG phonemes. Its callback
returns 1 to continue and 0 to stop. Decimal punctuation is preserved.

Activation selects pipeline mode, Nemotron/fr, Claude CLI and Piper/fr-FR at
speed 0.90. Streaming TTS is disabled for this initial profile so one complete
synthesis runs at a time. Existing microphone and playback devices are preserved.

## Activate the known BC-250 installation

Run as the normal desktop user:

```bash
cd /tmp/dodaios-piper-test &&
git pull --ff-only &&
~/.dodaios/.venv/bin/python -I scripts/enable_french_voice.py --apply
```

Without --apply, the script checks only. It requires the existing systemd user
service, eSpeak NG and a working Claude subscription login. It refuses unknown
installed module hashes, service EnvironmentFiles and conflicting effective
YAML/environment settings before stopping the service.

It backs up the five modified Python modules, active TOML and existing drift
baseline, preserves native libraries, resolves model paths like the real
launcher, reuses existing models and downloads missing speech bundles,
including permanent Siwis. With the service stopped it validates native
model loading and synthesis in the production interpreter, checks the real
STT/TTS factories and starts the service.

On failure it restores the backed-up files and attempts to restart the
previously running service. Downloaded model bundles remain for retries.
Restoration/restart failures are reported separately. Backups remain beside
the active configuration under french-voice-backups/.

Open Dodaios and start a voice conversation after activation. The script
does not record the microphone or claim end-to-end hardware acceptance.

## Validation and limits

- 306 targeted tests passed for language resolution, reply-language retention,
  Piper and activation/rollback, including callback-respecting native batching.
- New activation/Piper code passes Ruff. Existing manager/language modules have
  25 pre-existing Ruff findings, with no new findings introduced here.
- Full pytest collection is blocked at tests/contract/test_agent_autonomy.py
  by the development environment's missing mcp dependency.
- No live BC-250 execution was available during this implementation.
- Claude CLI uses the existing subscription and is slower to start than an API.
  Its conversational adapter is text-only and does not itself execute tools.
- No GPU clocks, voltages, drivers or native package versions are changed.
  Sustained whole-application memory use remains to be observed.

Sherpa 2 removed the embedded phonemizer:
https://github.com/k2-fsa/sherpa-onnx/issues/3731

Python callback contract for the runtime used on the BC-250:
https://github.com/k2-fsa/sherpa-onnx/blob/v2.0.1/sherpa-onnx/python/csrc/offline-tts.cc

External eSpeak NG and voice models retain their own licenses.
