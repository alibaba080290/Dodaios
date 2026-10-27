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

The browser path also maps French to fr-FR and protects the reply pin from a
browser UI-language hint. Registered model-less providers stay in the main
fallback chain, allowing Claude CLI to use its subscription's default model.
The headless launcher declares that it has no native speech pipeline: browser
audio initializes on demand, so the UI no longer waits for a nonexistent native
warm-up event.

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

It backs up the eight modified Python modules, active TOML and existing drift
baseline, preserves native libraries, resolves model paths like the real
launcher, reuses existing models and downloads missing speech bundles,
including permanent Siwis. With the service stopped it validates native
model loading and synthesis in the production interpreter, checks the real
STT/TTS factories and browser language route, and checks the actual brain tier
chain before requesting one short Claude subscription response with no tools.
It then starts the service and waits for its HTTP voice-readiness endpoint.
This request uses the existing control key inside an isolated worker, so an
enabled browser lock remains supported without exposing the key in arguments.

On failure it restores the backed-up files and attempts to restart the
previously running service. Downloaded model bundles remain for retries.
Restoration/restart failures are reported separately. Backups remain beside
the active configuration under french-voice-backups/.

Open Dodaios and start a voice conversation after activation. The script
does not record the microphone or claim end-to-end hardware acceptance.

## Validation and limits

- Targeted tests cover language resolution, reply-language retention, Piper,
  browser French routing, the model-less Claude chain, headless startup and
  activation/rollback, including callback-respecting native batching.
- New activation/Piper/browser code passes Ruff. Existing manager, launcher and
  language modules retain their baseline lint findings.
- Full pytest collection is blocked at tests/contract/test_agent_autonomy.py
  by the development environment's missing mcp dependency.
- Wider brain tests also have six pre-existing failures: the same-provider
  fallback in test_tier_model_resolution and five voice-key assertions in
  test_manager_voice_key_chain. All six were reproduced with the unchanged
  pre-fix manager, as well as with this change.
- The user confirmed the earlier native-model activation passed on the BC-250.
  The updated browser activation still requires execution there; microphone,
  playback and whole-application acceptance cannot be proved in this environment.
- Claude CLI uses the existing subscription and is slower to start than an API.
  Its conversational adapter is text-only and does not itself execute tools.
- No GPU clocks, voltages, drivers or native package versions are changed.
  Sustained whole-application memory use remains to be observed.

Sherpa 2 removed the embedded phonemizer:
https://github.com/k2-fsa/sherpa-onnx/issues/3731

Python callback contract for the runtime used on the BC-250:
https://github.com/k2-fsa/sherpa-onnx/blob/v2.0.1/sherpa-onnx/python/csrc/offline-tts.cc

External eSpeak NG and voice models retain their own licenses.

The six independently reproduced baseline assertion failures are:

- `test_router_fallback_chain_still_offers_pro_as_failover`
- `test_voice_key_leaves_the_turn_chain_while_another_provider_remains[fast]`
- `test_voice_key_leaves_the_turn_chain_while_another_provider_remains[deep]`
- `test_voice_key_primary_hands_the_turn_to_another_family`
- `test_tool_lead_prefers_a_family_off_the_voice_key`
- `test_scheduled_routine_chain_skips_the_voice_key`
