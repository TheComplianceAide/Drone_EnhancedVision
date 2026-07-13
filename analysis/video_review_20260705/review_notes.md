# 2026-07-05 Flight Recording Review Notes

Reviewed all extracted 5-second contact sheets from the five local screen recordings in `recordings/`.

## Recordings

- `mac_screen_20260705_121112.mov` - 0.17 min.
- `mac_screen_20260705_121157_10min.mov` - 10.00 min.
- `mac_screen_20260705_122938_20min.mov` - 20.00 min.
- `mac_screen_20260705_124955_20min_continued.mov` - 2.89 min.
- `mac_screen_20260705_143116_20min_continued.mov` - 20.00 min.

## Main Findings

- Launcher and browser windows repeatedly competed with the actual flight windows. The mission windows need to own the cockpit once launched.
- Boat/wake footage was the strongest positive signal. The system can find useful lake activity, but the overlay can obscure boats/wakes and the auto crop sometimes follows wake texture instead of the leading object.
- Land, trees, roofs, decks, railings, and grass triggered too much edge/trail energy. This looked impressive but was often semantically weak.
- Max/NightVision crop ergonomics were much better than the older small microscope panes. The risk is that extreme zoom on soft source pixels can look more confident than the evidence supports.
- EventScope/ISR/LakeHouse auxiliary panes were sometimes black or waiting while reacquiring. During flight that reads as dead air; a live scout crop is more useful.
- Stable target identity matters. Fast target switching made some modes feel reactive instead of deliberate.
- The last `143116` recording was mostly static Codex/Finder review state, not new flight footage. It still reinforced the need for window hygiene, but it did not add new vision-enhancement evidence.

## Mode-Specific Notes

- Lucky Skyline / Max: useful for structures, docks, roofs, and distant detail when the source crop still has enough pixels. Needs an on-screen detail honesty meter so operators know when to widen zoom or move closer.
- Radar Motion: target choice should stay sticky unless a better target clearly wins. AutoZoom should label center-scout mode when no confirmed target exists.
- Temporal EventScope: black waiting microscope pane should become a center scout crop. Overlay opacity should drop under camera motion or high-texture clutter.
- ISR Recon Suite: same reacquire/scout crop problem as EventScope; keep the auxiliary pane informative even without a selected pulse.
- LakeHouse AutoScout: best lake/wake mode, but needs stickier target selection, slight lead on wave targets, ranked target visibility, and scene-aware overlay damping when the estimated water band is actually land/vegetation texture.

## Changes Made From Review

- `app_Launcher_v2.py`: launcher iconifies shortly after mission launch so it does not cover the cockpit.
- `_12_M5_NightVision_Max_Rev1.py`: added detail/usable-source scoring, HUD/proof-panel labels, AI blend damping for weak evidence, and snapshot metadata.
- `_08_M5_LuckySkylineSuperZoom_Rev2.py`: added detail/usable-source scoring to live and zoom HUDs.
- `_09_M5_TemporalEventScope_Rev2.py`: added sticky selection, scout fallback crop, and adaptive overlay opacity.
- `_10_M5_ISR_ReconSuite_Rev2.py`: added sticky selection, scout fallback crop, and adaptive overlay opacity.
- `_11_M5_LakeHouse_AutoScout_Rev2.py`: added sticky selection, wake lead, scout fallback crop, top-3 target summary, and adaptive lake overlay opacity.
- `_07_Radar_Motion_GPU_AutoZoom_Rev1.py`: smoothed track updates, added sticky AutoZoom target choice, and labeled center-scout fallback.
