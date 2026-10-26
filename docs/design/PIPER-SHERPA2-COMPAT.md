# Piper / sherpa-onnx 2.x compatibility (WIP)

## Reproduction (CachyOS x86_64, Dodaios 4.0.1)

- `sherpa-onnx` and `sherpa-onnx-core`: 2.0.1.
- Voice directories: ONNX + `tokens.txt` + `espeak-ng-data`, without `lexicon.txt`.
- Built-in German voice and English voice fail to produce audio.
- Native log: `phoneme_codepoints is empty, no lexicon, and no frontend available`.
- Nemotron STT works in the same runtime; downgrading sherpa globally is unsafe.
- The four built-in Piper voices do not include French.

## Verified upstream change

sherpa-onnx 2.0.0 removed its embedded eSpeak/piper-phonemize frontend:
https://github.com/k2-fsa/sherpa-onnx/issues/3731

The legacy `vits.data_dir` is deprecated/ignored in 2.x. An external
phonemization frontend, passed phoneme codepoints, or a compatible lexicon is
required before audio generation can work.

## Scope of this initial draft

- Fail early with a descriptive message if sherpa-onnx 2.x is paired with a
  legacy Piper bundle without a lexicon.
- Do not report success on a silent generated turn.
- Regression test for silent synthesis.

## Remaining before functional fix

1. Evaluate sherpa 2.x Python generation API for phoneme-codepoint input;
   select a properly licensed, packaged multilingual frontend.
2. Support French `fr_FR` voice artifacts with verified URLs, checksums,
   licensing, installation/update flows, and correct model catalog selection.
3. Validate a French real-audio smoke test and other existing languages
   under sherpa 2.0.1.
4. Run tests and lint in CI and test on the target BC-250 without changing
   the functioning Nemotron runtime.
5. Measure sustained memory use and ensure speech models are released on
   provider switches.

**This draft is not yet a working French Piper solution. Do not merge or
deploy it until the missing integration and tests are completed.**
