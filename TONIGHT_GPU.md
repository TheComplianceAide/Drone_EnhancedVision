# Tonight's GPU setup

Open **Start_Tonights_GPU_Launcher.command**. The regular launcher uses the same updated configuration.

- **Fable Motion ISR V3:** MPS GPU detection frontend and native Metal temporal-state processing, with the current SEARCH preset, track history, night preview and target inspection. GPU is required; a backend failure stops the mission visibly instead of silently using CPU.
- **NightVision Max V3:** MPS reconstruction is required. Full-frame night preview is available immediately; the selected region improves only when its source-support gates accept observations. The optional extra temporal denoiser uses CPU alongside GPU reconstruction.
- **Fable SuperRes V4:** MPS reconstruction is required; the extra temporal preview uses CPU alongside it. Source ROI selection supports 2–16x digital magnification and raw/CLEAR inspection. It cannot recover absent optical detail.
- **ImageScout V3:** enhanced inspection and night profiles, with MPS temporal quality when available.

Use **V** for the night preview in Motion/NightVision/SuperRes, **N** for ImageScout's night profile, **I** for detailed inspection, and **T** for temporal quality. Brackets adjust inspection zoom; 4/6/8/2 pan in the inspection pane. **R** resets source history; **Q** quits.

All updates are included. Temporal quality is available on demand; it is not enabled automatically because the measured full1080p eight-frame mode costs roughly195ms/frame. Raw comparison stays available, and the normal source stream keeps latest-frame semantics. GPU-required means the substantial supported stages use the GPU; decoding, registration, tracking, GUI and some denoising still use CPU.

**Experimental Rev5 remains separately listed.** It has controlled detector gains but still fails broad false-confirm/tracking acceptance. The GPU field selection is Rev3. Neither a local RTMP rehearsal nor a GPU badge proves native night-flight range or whole-flight recall.

The shared live capture helper now applies the same short FFmpeg probe options when opening any network stream and restores the environment afterward. This fixes the NightVision startup retry observed in the 1080p local RTMP rehearsal; local video files do not receive those live-only defaults.

[GPU launch and live RTMP rehearsal receipt](analysis/flight_review_20260714/gpu_tonight_20260904_e9021912/README.md).
