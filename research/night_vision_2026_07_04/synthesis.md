# Night Vision Research Sweep - 2026-07-04

## Scope

This pass used 2024-07-04 through 2026-07-04 as the "last two years" window.

- arXiv candidate harvest: 429 papers / technical reports.
- Selected paper set: 180 title/abstract-level reads focused on low-light, video, denoising, dehazing, underwater, event-guided, RAW, efficient/mobile, and UAV-adjacent enhancement.
- GitHub candidate harvest: 552 unique repositories from low-light, realtime, Retinex, Zero-DCE, underwater, dehazing, denoising, ONNX, Core ML, and video-enhancement searches.
- Selected GitHub set: 100 repositories.
- README pass: 80 repositories attempted, 75 README fetches succeeded.

Generated corpus files:

- `arxiv_papers.json`
- `selected_papers_180.json`
- `github_repos_100.json`
- `github_repos_80_readmes.json`

## What The Research Says

### 1. The strongest live-video win is temporal, not single-frame brightening

Recent low-light video and RAW-video work keeps circling the same point: use neighboring frames. The AIM 2025 low-light RAW video denoising challenge frames the task around temporal redundancy, exposure-time limits, and sensor-specific signal-dependent noise. This directly supports our current direction: align frames, combine them, and avoid treating every RTMP frame as an isolated photo.

Practical implication for the MacBook:

- Keep a rolling frame buffer.
- Stabilize the selected ROI.
- Use robust temporal fusion before heavy enhancement.
- Reject/weight frames by alignment quality and motion.
- Use the deep enhancer on the temporally cleaned crop, not on the whole live frame.

### 2. Efficient/mobile LLIE is now a primary research track

The NTIRE 2026 efficient low-light challenge is specifically about mobile deployability under limited compute. The broader NTIRE 2026 LLIE challenge also emphasizes joint denoising and low-light restoration. This matters because our target is not a cloud render; it is a live MacBook field console.

Practical implication:

- Prefer lightweight ConvNet / Retinex / LUT / small transformer paths.
- Avoid full-frame diffusion in the live loop.
- Use a quality governor that dynamically shrinks inference resolution or cadence when FPS drops.

### 3. Retinex is still useful, but only with noise awareness

Retinex-style decomposition remains common in modern papers and repos: Retinexformer, Reti-Diff, UDU, TempRetinex, Poisson-informed Retinex, and several 2025-2026 flow/diffusion variants. The useful idea is not "make illumination brighter"; it is "separate illumination from reflectance/detail, then avoid amplifying shadow noise as if it were structure."

Practical implication:

- Our current CLAHE/sharpen steps are too blunt.
- Add a signal/noise map for the ROI.
- Apply stronger lift only where the temporal stack says there is repeatable structure.
- Suppress isolated high-frequency noise before sharpening.

### 4. LUT-based enhancement is a serious realtime candidate

FastLLVE, WaveLUT, MobileIE, and related repos point toward lookup-table or learned lightweight transforms for real-time image/video enhancement. LUT methods are attractive for Apple Silicon because they can be fast, predictable, and potentially portable to Core ML/Metal.

Practical implication:

- Prototype a lightweight learned/LUT enhancer as the "fast pass."
- Keep IAT or Retinexformer-like models as a slower "detail pass" on ROI.
- Long-term, convert the winning fast pass to Core ML.

### 5. Event-camera papers validate our EventScope idea, even without event hardware

The event-guided low-light papers are not directly drop-in because we only have RGB video, not a true event sensor. But they strongly validate the conceptual direction: low-light frame video struggles with temporal correspondence, while high-temporal-resolution brightness-change information helps deblur and track structure.

Practical implication:

- Keep the EventScope trail/motion microscope.
- Make it feed the ROI selector and stack weighting.
- Treat frame-difference events as a weak event sensor.
- Use event heat only for targeting and temporal confidence, not as final "pretty video."

### 6. Underwater/dehazing work is relevant, but not universally safe

Underwater and dehazing methods address scattering, veiling, color absorption, and low contrast. Those ideas help haze, fog, lake mist, rain, and washed-out city glow. They can hurt true darkness by turning noise into gray haze.

Practical implication:

- Keep dehaze as an auto-detected profile, not always-on.
- Use haze/veiling metrics before applying dark-channel or transmittance-style correction.
- In true night scenes, prioritize denoise/temporal fusion before dehaze.

### 7. Diffusion/flow models are quality leaders, not first-choice live tools

Recent diffusion, rectified-flow, and flow-matching papers are very relevant for image quality, especially offline or single-frame enhancement. They are generally too heavy for full-frame live use on this MacBook unless distilled, downscaled, or used on still snapshots.

Practical implication:

- Do not put diffusion in the live full-frame loop.
- Consider it for snapshot polish, offline comparison, or teacher-model distillation.
- If we adopt a generative method, confine it to paused ROI snapshots and label it clearly.

## GitHub Signals Worth Borrowing

Most useful implementation lanes found:

- Core ML/iOS realtime low-light: `LeiGitHub1024/iOSRealTimeLowLightVideoEnhancement`
- Realtime temporal video: `Wenhao-Li-777/FastLLVE`, `zkawfanx/StableLLVE`, `xiaogang00/LLVE_STCD`
- Event-guided video: `intelpro/ELEDNet`, `sherrycattt/EvLowLight`, `YuXie1/CompEvent`
- Modern LLIE baselines: `caiyuanhao1998/Retinexformer`, `Fediory/HVI-CIDNet`, `JIA-Lab-research/SNR-Aware-Low-Light-Enhance`, `Li-Chongyi/Zero-DCE`
- Edge/mobile/export examples: `AVC2-UESTC/MobileIE`, `hpc203/Low-Light-Image-Enhancement-onnxrun`, `hpc203/low-light-image-enhancement-opencv-dnn`, `Kazuhito00/LYT-Net-ONNX-Sample`, `Kazuhito00/Retinexformer-ONNX-Sample`
- Underwater/dehaze ideas: `suhas-srinath/undive`, `xahidbuffon/FUnIE-GAN`, `Tian-Yu-CAO/DNnet-A-Lightweight-Network-For-Real-Time-4K-Underwater-Image-Enhancement`, `wangyanckxx/Single-Underwater-Image-Enhancement-and-Color-Restoration`

Repo signal counts from the 80-README sample:

- lowlight: 59
- realtime: 54
- temporal: 52
- mobile/edge: 29
- LUT: 24
- underwater/dehaze: 12
- model export: 10
- diffusion: 6

## Recommendation For Our Next Significant Mac Improvement

Build `_12_M5_NightVision_Max_Rev1.py` as a proof-oriented fusion viewer, not as another one-off enhancement script.

Core pipeline:

1. Decode RTMP with `LatestFrameGrabber`.
2. Full-frame display uses cheap OpenCV enhancement only.
3. User clicks/taps an ROI, or EventScope auto-selects one.
4. Maintain a rolling ROI buffer of 12-40 frames.
5. Estimate alignment with phase correlation first, optical-flow fallback later.
6. Reject frames with weak alignment response or too much local motion.
7. Fuse aligned frames with robust temporal averaging / median-clipped accumulation.
8. Build a confidence/noise map from temporal agreement.
9. Apply Retinex/LAB/CLAHE only where the confidence map says signal exists.
10. Run IAT or another MPS model only on the fused ROI, not full-frame.
11. Add an optional fast LUT/model path for realtime preview.
12. Show four panes: raw crop, current Rev5-style crop, stacked crop, stacked+AI crop.
13. Save side-by-side snapshots and metrics for every test clip.

Expected material improvement:

- High: static or slow targets, skyline/shoreline/roofline, docks, boats, roads, glints, lights, hazy but not black scenes.
- Medium: moving targets if the ROI is tracked and the stack is short.
- Low: fast gimbal motion, black sky with no photons, crushed compression, strong motion blur, tiny targets below sensor/RTMP resolution.

## Implementation Sequence

Phase 1: prove with no new model.

- Build stacked ROI viewer.
- Compare raw vs current Rev5 vs stacked-only.
- Add side-by-side snapshot export.

Phase 2: add the existing IAT model correctly.

- Run IAT only on 320-480 px ROI.
- Use MPS when it wins; fall back to CPU gracefully.
- Add quality governor for ROI inference size/cadence.

Phase 3: add fast model/export path.

- Try LYT-Net / MobileIE / FastLLVE-style path.
- Prefer ONNX or PyTorch first for evaluation.
- Convert only the winning candidate to Core ML.

Phase 4: field hardening.

- Scene classifier: true night vs haze vs city glow vs water.
- Stack reset/reacquire logic.
- Artifact warnings when enhancement is hallucinating/noise-amplifying.
- Auto-snapshot evidence pack.

## Bottom Line

The research backs the direction. The biggest next improvement is not a bigger single-frame enhancer; it is a temporally aligned ROI enhancement pipeline with noise-aware fusion, fast preview enhancement, and optional heavy AI detail on the crop. That matches both the current literature and the measured M5 constraint that full-frame IAT is too slow while crop-level IAT is practical.
