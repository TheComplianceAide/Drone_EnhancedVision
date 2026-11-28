# AstraOct26 publication and preflight review workflow — October 16, 2026

Status: prepared for GitHub review; this receipt does not assert a merge or flight acceptance.

The project now requires a short playable demonstration of actual script behavior for operator-visible vision changes. The owner reviews detections, misses, false cues, enhancements and target loss while on the ground, then uses the same replay to iterate. The rule is in `AGENTS.md` and introduced in the root README. Demos keep text minimal, include raw comparisons and representative failures, and distinguish recorded replay from flight evidence.

This publication extends the existing GPU/launcher foundation at `1a00b415a210a7670b40ee1d58f8e5811ecfd9ca` with the current experimental AstraOct26 application, its pinned model setup, MPS-required launcher integration, Follow fix, tests, documentation and selected evidence. It preserves the foundation's experimental acceptance limits. Unrelated local changes, recording deletions, raw flight recordings, model weights and private chat/research inventories are excluded. The separate October 5 Click-to-Zoom repair is not part of this publication.

## Validation of the publication copy

- Compilation passed for the Astra application, runtime, models, decoder, validator, launcher and tests.
- 175 of 176 unit tests passed. The sole failure is `test_file_decoder_preserves_nonuniform_pts_and_prompt_close`, where `source.ended` remains false after five seconds. This also failed before the Follow fix. The shared checkout previously ran 179 tests; the three unrelated Click-to-Zoom tests account for the publication copy's smaller total.
- All six new Follow regressions and the MPS-required launcher policy tests pass. Application/runtime/model/decoder/launcher source hashes match the tested shared checkout snapshot. The original Motion Rev3 detector and shared GPU/operator modules match the existing PR foundation.
- The same-input Follow regression used 780 delivered native 1920x1080 frames. In the fixed negative window, false driveway-follow frames decreased from 183 to zero. The candidate still loses the child; continuous identity tracking is not established.
- The saved actual GUI demo decodes completely, keeps LOST visible, and records successful detector/research/writer shutdown. It is an FFmpeg-PTS replay demonstration with default labels/Align; optional-model cold-load shutdown and the field file-decoder failure remain open.
- Existing baseline and source decode warnings are retained. This is not all-green validation, whole-flight recall, ball/bird detection acceptance, aircraft-link proof or flight readiness.

The [Follow receipt](../astraoct26_follow_fix_20261016_134545_ce95b8a8/README.md) contains the exact commands, original code/source hashes, raw comparisons, intermediate failures and video. `source_snapshot.json` lists this publication's explicitly selected source/evidence files; `validation.json` and `unit_tests.log` record the publication checks. These checks reuse the repository Python 3.11 environment without changing its packages.

```sh
env DRONE_VISION_NO_RELAUNCH=1 /Users/randyblasik/Documents/Drone_EnhancedVision/.venv/bin/python -m py_compile _AstraOct26.py astra_oct26.py astra_oct26_runtime.py astra_oct26_decode.py astra_oct26_models.py astra_oct26_validation.py app_Launcher_v2.py m5_field_launch.py tests/test_astra_oct26.py
env DRONE_VISION_NO_RELAUNCH=1 /Users/randyblasik/Documents/Drone_EnhancedVision/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

GitHub branch protection requires one approving review and signed commits. The publication process preserves those settings; any resulting merge status must be read back from GitHub separately.
