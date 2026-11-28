# OSW26TZ04-NV001 fit review

- Reviewed: 2026-08-12
- Repository baseline: [`25ac9a3`](https://github.com/TheComplianceAide/Drone_EnhancedVision/tree/25ac9a3df478f4935c6a9927516589e35500ce76)
- Decision: **Applicable as a supporting technical seed; not applicable as a stand-alone or as-is solution.**

## Executive decision

Do **not** submit `Drone_EnhancedVision` as the proposed solution by itself. The repository demonstrates useful single-UAS electro-optical video enhancement, stabilized motion detection, temporal tracking, super-resolution, and operator decision-support building blocks. The topic's evaluated center of gravity is materially different: an open-weight foundation-model system that collaborates across multiple platforms and modalities, resolves identity across views, maintains a predictive world model, and operates under constrained communications and degraded PNT.

The defensible posture is:

- **No-bid as ComplianceAid alone or as a repackaging of this repository.**
- **Conditional pursuit only** if a qualified U.S. nonprofit research institution and senior foundation-model/multi-platform sensor-fusion personnel are already available to own the missing research core and execute a signed teaming/IP agreement before the deadline.
- If pursued, position `Drone_EnhancedVision` only as one existing EO-UAS edge node and as a source of temporal-registration, tracking, enhancement, and synthetic-test components.

No research-institution partner, signed agreement, or cross-platform/foundation-model prototype was supplied or found in this repository during this review. That is an evidence gap, not a claim that none exists elsewhere.

## Authoritative opportunity facts

The attached text is a solicitation topic, not a completed proposal. The controlling sources reviewed were the official [Release 4 full BAA](https://www.dodsbirsttr.mil/submissions/api/public/download/solicitationDocuments?documentType=RELEASE_INSTRUCTIONS&release=4&solicitation=DOD_STTR_2026_P1_CTZ), [BAA preface](https://www.dodsbirsttr.mil/submissions/api/public/download/solicitationDocuments?documentType=RELEASE_PREFACE&release=4&solicitation=DOD_STTR_2026_P1_CTZ), [topic record](https://www.dodsbirsttr.mil/topics/api/public/topics/cba54482576b46ddaa02a7ee564fa11a_86579/details), and all 15 published [topic Q&A responses](https://www.dodsbirsttr.mil/topics/api/public/topics/cba54482576b46ddaa02a7ee564fa11a_86579/questions).

| Gate | Official requirement |
| --- | --- |
| Program | DoW 2026 STTR BAA, Release 4; topic `OSW26TZ04-NV001` |
| Close | **2026-08-19 at 12:00 p.m. ET**; complete proposal must be certified and submitted in DSIP |
| Q&A | Scheduled to close to new questions on **2026-08-12 at 12:00 p.m. ET** |
| Phase I ceiling | Base amount must not exceed **$314,363** |
| STTR teaming | Small business must perform at least 40% and one U.S. nonprofit research institution at least 30% of the project |
| Mandatory topic attachments | Signed STTR Cooperative Research and Development Agreement plus an IP/data-rights allocation statement between the small business and research institution |
| Phase I technical minimum clarified in Q&A | At least two collaborating platforms (physical or simulated), at least two distinct modalities, and an architecture based on adaptable open-weight/pre-trained foundation models |
| Data | No Government-furnished data, models, or telemetry for Phase I; offeror must source or simulate data and document provenance, licensing, and export-control compliance |
| Phase II destination | At least three heterogeneous platforms spanning multiple domains, all listed KPPs treated as firm thresholds, and transition planning across at least two Services |
| Other controls | Projected CMMC Level 2 (Self); ITAR/EAR restrictions; foreign-affiliation and foreign-national disclosures |

The evaluation instructions give particular weight to the foundation-model-based collaborative AiTR approach; the foundation-model, autonomy, and Army-relevant fusion qualifications of both the small business and research institution; and tri-Service transition potential. Those factors make a strong research partner central, not administrative window dressing.

## What the repository genuinely contributes

The current repository is a public, Mavic-centered video-processing toolkit. Its strongest relevant evidence is:

- One RTMP/RTSP/video EO feed from a DJI Mavic 3 or equivalent source, documented in the [README](https://github.com/TheComplianceAide/Drone_EnhancedVision/blob/25ac9a3df478f4935c6a9927516589e35500ce76/README.md#L33-L46).
- Ego-motion compensation, registered residuals, track-before-detect temporal integration, Kalman tracks, persistent IDs, and optional YOLO chip labeling in [MotionISR](https://github.com/TheComplianceAide/Drone_EnhancedVision/blob/25ac9a3df478f4935c6a9927516589e35500ce76/_09_M5_Fable_MotionISR_Rev1.py#L1-L61).
- Motion-compensated low-light enhancement with an IAT learned enhancer and a classical fallback in [NightVision](https://github.com/TheComplianceAide/Drone_EnhancedVision/blob/25ac9a3df478f4935c6a9927516589e35500ce76/_10_M5_Fable_NightVision_Rev1.py#L1-L45).
- Multi-frame registration, reconstruction, and optional Real-ESRGAN enhancement in [SuperRes](https://github.com/TheComplianceAide/Drone_EnhancedVision/blob/25ac9a3df478f4935c6a9927516589e35500ce76/_11_M5_Fable_SuperRes_Rev1.py#L1-L54).
- Operator-in-the-loop target lock, reacquisition, event recording, and mission-report generation in [Overwatch](https://github.com/TheComplianceAide/Drone_EnhancedVision/blob/25ac9a3df478f4935c6a9927516589e35500ce76/_12_M5_Fable_Overwatch_Rev1.py#L1-L61).
- Offline model-weight use and CPU/MPS fallback behavior, useful for a future edge-compute baseline.

These are credible components for **one visible-light UAS sensor node**. They do not prove the collaborative system requested by the topic.

## Requirement-to-evidence comparison

| Topic requirement | Current repository evidence | Fit |
| --- | --- | --- |
| Open-weight VLM, VLA, SSM, or hybrid foundation-model core | No CLIP, LLaVA, Florence, Qwen-VL, OpenVLA, RT-2, S4/S5, or Mamba implementation found. IAT is an illumination-enhancement transformer, not a vision-language foundation model. YOLO is a conventional CNN detector. | **Missing** |
| Collaborative multi-platform inference | The active Fable tools consume one video source; no platform-to-platform collaboration or feature exchange was found. | **Missing** |
| At least two Phase I sensor modalities | Visible EO processing is present. Low-light enhancement of the same RGB stream is not a distinct IR, acoustic, SAR, LiDAR, or RF modality. | **Missing** |
| Cross-platform target correspondence and handoff | Persistent IDs and reacquisition exist within one stream; no cross-platform identity resolution or handoff metric is implemented. | **Partial seed** |
| Geometry-consistent multimodal fusion | Single-stream image registration is useful prior art, but no cross-sensor calibration or fusion exists. | **Partial seed** |
| Predictive spatiotemporal world model | Constant-velocity Kalman tracking is present; no shared world model or 30-second/2-meter forecast evidence exists. | **Partial seed** |
| Blue/Red/Gray classification | No friend/foe/neutral taxonomy or evaluation was found. | **Missing** |
| Degraded-PNT and degraded-comms operation | No probabilistic localization uncertainty, link simulation, bandwidth protocol, or PNT-denial test was found. | **Missing** |
| Agentic autonomy and documented APIs | Auto-lock/task recommendation concepts are adjacent, but there is no multi-platform autonomy API or software-in-the-loop orchestration contract. | **Partial seed** |
| Tactical-edge compute and latency | CPU/MPS paths and governors exist, but the full collaborative pipeline is absent and therefore cannot be benchmarked against the sub-500 ms KPP. | **Partial seed** |
| Model integrity, adversarial robustness, provenance, explainability | Some third-party notices and offline weights exist. No complete model cards, dataset supply-chain record, adversarial evaluation, poisoning defense, or output-explanation design was found. | **Material gap** |
| Phase II multi-domain transition | No UGV/USV integration, transition stakeholder evidence, or three-platform field plan was found. | **Missing** |

## Executed validation and blind spots

The existing code was exercised in an isolated Windows Python environment without changing repository source. Results:

| Check | Result |
| --- | --- |
| `python m5_v2_validation.py` | **PASS**; all four M5 synthetic validation gates passed |
| `python _09_M5_Fable_MotionISR_Rev1.py --selftest` | **PASS**; synthetic mover coverage 1.0, ego-compensated confirmed false positives 0 versus 37.7786/frame with compensation off, and track-before-detect recall 1.0 versus 0 off |
| `python _10_M5_Fable_NightVision_Rev1.py --engine classical --selftest` | **PASS**; synthetic classical CPU path improved PSNR and passed registration/adaptation checks |
| `python _11_M5_Fable_SuperRes_Rev1.py --selftest` | **PASS**; synthetic CPU reconstruction checks passed |
| `python _12_M5_Fable_Overwatch_Rev1.py --selftest` | **PASS**; 32 synthetic tracking, lock/reacquisition, DVR, briefing, and adaptation checks passed |

Blind spots: these were synthetic, single-stream, CPU/classical-path checks. Apple MPS, the learned IAT path, YOLO inference, a live Mavic feed, multiple platforms, multiple modalities, cross-platform target identity, degraded PNT/comms, adversarial robustness, and every topic KPP remain unvalidated. A pass here establishes repository health for selected modules; it does **not** establish topic feasibility.

## Additional proposal-readiness issue

GitHub reports the repository as public, but no root project license was found and GitHub reports `licenseInfo: null`. Publicly visible source is not automatically open-source-licensed. Before sharing it as an open-source contribution or allocating rights with an STTR partner, the project needs an intentional first-party license/data-rights position plus a full inventory of third-party code, weights, model provenance, and permitted government/partner use. That is separate from, and does not replace, the mandatory STTR IP/data-rights agreement.

## Go/no-go decision rule

Reverse the current no-bid posture only if all of the following can be evidenced immediately:

1. A qualified U.S. nonprofit research institution will sign the required cooperative R&D and IP/data-rights agreements before submission.
2. Named key personnel can credibly lead foundation-model adaptation, multi-platform autonomy, and geometry-consistent multi-modal fusion.
3. The team can propose a six-month Phase I prototype with at least two simulated collaborating platforms and two real or simulated modalities, using an open-weight model and baseline plans for every Phase II KPP.
4. The team can source/license the data and models and document model cards, labeling controls, supply-chain provenance, export controls, and adversarial robustness.
5. SBA Company Registry, DSIP firm/user access, SAM/award eligibility, CMMC Level 2 self-assessment readiness, and the seven required proposal volumes are verified live.
6. The proposal presents `Drone_EnhancedVision` honestly as a reusable EO edge-node baseline, not as evidence that the requested collaborative foundation-model system already exists.

If any of gates 1-3 is unavailable now, the practical recommendation is **no-bid for this release** and preserve the repository as a starting point for a future partnered topic.

## Evidence integrity

- Official full BAA PDF SHA-256: `438d4ac09094ea870c9287aec5e5c59f721ed5b62e14bdbc666a072fdf7d9610`
- Official BAA preface PDF SHA-256: `0e13d7d179f929f751cc63698065b3efddef7aa49593317f40825a1cc3464809`
- Official topic-details response SHA-256 at review: `1ad2debdd283469e9049d27d9814788b28317146a6ea292c546cedad9ba9f702`
- Official 15-response Q&A payload SHA-256 at review: `dd2dee59613105c1a03e767b5a547de0e880673023103ff0a2b3f45dae40bdcb`
- Repository baseline commit: `25ac9a3df478f4935c6a9927516589e35500ce76`
