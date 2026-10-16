# September 4 M5 night vision, ISR and zoom update

Status: **PASS_NON_RELEASE**. This is the compact GitHub publication receipt for the local upgrade, evaluated September 4, 2026. It does not promote the experimental Motion Rev4 or clear native-flight acceptance.

The four featured apps add full-frame night display, larger raw/enhanced inspection, stable target selection and event history, and extended SuperRes ROI controls. A fused native Metal state update repairs and accelerates the Motion GPU engineering lane. [Operator controls](../../../OPERATOR_UPGRADES.md) describe the exact behavior and limits.

## Provenance and runtime

[Candidate hashes](candidate_sha256.json) bind all 16 upgraded code files. [Baseline hashes](baseline/hashes.json) and the preserved [baseline source archive](baseline-source.tar.gz) capture the pre-upgrade local comparison code, which was already different from GitHub main. Root Rev1 files are deliberately unchanged by this publication. Extract the archive in a separate directory and replay its frozen baseline when reproducing the historical comparisons. Source segment SHA-256, decoded PTS, ROIs, frame counts, thresholds, commands, and code provenance are embedded in each machine receipt below; the canonical catalog remains `testdata/flight_scenes/2026-07-14.json`.

[Hardware](hardware.json): Apple M5, 10 CPU cores, 10 GPU cores, 24 GiB unified memory; macOS 26.1 arm64, Python 3.11.14, PyTorch 2.10.0, OpenCV 4.13.0. Additional service cost: $0. The GUI and benchmark runs used local derived footage or synthetic inputs, not live DJI RTMP.

## Results and retained failures

| Lane | Result | Evidence |
| --- | --- | --- |
| Clean publication checkout | 132 tests passed; injected error-path messages are expected | [Test log](publication-tests.log) |
| Motion built-in self-test | S1–S14 pass, including repaired MPS parity | [Full self-test](final-tail-1.log) |
| Native Metal frontend | About 22% lower median 1080p processing time than CPU; fused and corrected eager MPS state outputs match exactly | [GPU timing distributions and hashes](gpu-benchmark/gpu-heavy-benchmark.json) |
| Spatial night preview | Three flight-derived low-light proxies pass bounded metrics, review required | [Preview receipt](night-preview/receipt.json) |
| NightVision MPS | PASS_METRICS_REVIEW_REQUIRED; 0 failures, 3 scope warnings | [NightVision receipt](nightvision/nightvision_rev3_validation.json) |
| SuperRes direct Rev1 A/B | PASS_METRICS_REVIEW_REQUIRED; 0 failures, 2 short-fixture warnings | [Direct receipt](superres-ab/superres_ab_validation.json) |
| SuperRes independent source honesty | PASS_METRICS_REVIEW_REQUIRED; 0 failures, 0 warnings | [Independent receipt](superres-independent/superres_v3_validation.json) |
| Motion/ImageScout source smoke | PASS_NON_RELEASE; candidate ground truth and transition acceptance still missing | [Source smoke](source-smoke.json) |
| Experimental Motion Rev4 | FAIL, all 10 failing gates retained; no promotion | [Experimental failure](motion-rev4-experimental-failure/motionisr_rev4_validation.json) |

[Summary](receipt.json) preserves all warnings and open gates. The launcher continues to pass `--device cpu` to Motion; explicit GPU evaluation uses `Start_MotionISR_GPU_Experimental.command`. NightVision and SuperRes retain their separately checked MPS restoration paths. No claim is made about GPU saturation, live-stream FPS, whole-flight recall, native night-flight range, or recovered identifying detail. Digital magnification does not increase optical resolution; higher ROI divisors and 3x reconstruction remain outside the bounded 2x flight evidence.

## GUI amendment and visual review

The full SuperRes quality runs bind the pre-label file SHA `003c35ee485743dae7d3732a308c26739bb150ab2a4d063252e8fcb939a23d5d`. The final file SHA is `1e81e58d6dc0dd15359bba8469d1ceba99cb99f9ba0c2b57265667d2e4f7ea8d`. A single GUI render call now accurately labels the registered input grid. [Amendment audit and real GUI replay](gui-label-amendment.json), [exact patch](gui-label-amendment.patch), and [preserved pre-label source](history/superres-rev3-before-label-amendment.py) document the distinction; reconstruction functions are AST-identical.

Original-resolution visual inspections and native OpenCV reset, zoom, night display, freeze, and quit checks were completed during the local task. [Visual review index](visual_review.json) records that review. This compact publication omits the large image/video artifacts; the original immutable 277 MB local package retains those images. Machine receipts retain their original local paths, so those paths are provenance references, not downloadable GitHub artifacts. No raw flight recordings are added here, and a clone alone cannot rerun flight acceptance without the hash-matching source corpus.

## Reusable commands

Use the repository Python 3.11 virtual environment with the versions above for the recorded environment. Run `npm ci` for the pinned Node Media Server 4.2.8 launcher dependency. Validation outputs must use new directories.

```bash
env DRONE_VISION_NO_RELAUNCH=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python m5_motion_gpu_validation.py --output-dir /tmp/motion_gpu_new_run
.venv/bin/python m5_flight_catalog.py --verify-sources --hash
```

[Exact final self-test and benchmark commands](final-tail-commands.json) and [replay commands](final-replay-commands.json) retain every full command. `AGENTS.md` documents the acceptance lanes; historical July receipt references outside this compact package may remain local. [Publication manifest](publication-manifest.json) identifies the exact files included here, independently of the larger local artifact manifest.
