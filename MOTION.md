# MOTION.md — Anti-Slop 3D Camera & Editorial Motion System

Before designing, generating, or animating any video or motion graphic in this workspace, read this file in full. Every colour, font, camera rig, spring physics curve, audio rule, and QA step comes from this specification.

---

## 1. Strict Matte Palette (Zero Gradients, Zero Glows, Zero Drop Shadows)

| Token | Hex | Role |
|---|---|---|
| `BG` | `#121214` | Flat matte obsidian canvas background |
| `CORAL` | `#FF5A5F` | Primary accent: active spoken word, hero stat cards, salient 3D weight blocks, ceiling markers |
| `FG` | `#EAEAEA` | Primary off-white: headlines, protected 4-bit logic blocks, primary callout frames |
| `MUTED` | `#8A8A96` | Secondary technical labels, camera telemetry, inactive metadata |
| `GRID` | `#26262B` | 2px structural Swiss grid rules, 3D perspective floor grid, and wireframe envelopes |
| `MONUMENT_STROKE` | `#5E5E6E` / `#FF5A5F` | Giant 3D world-space background typography strokes |

- **Directional 3D Face Shading (Flat Matte Only):**
  - White 3D blocks: Top `#EAEAEA`, Front `#CFCFD8`, Side `#9E9EA8`, Bottom `#70707A`, Stroke `#121214`.
  - Coral 3D blocks: Top `#FF5A5F`, Front `#E0484D`, Side `#B83539`, Bottom `#8F2428`, Stroke `#121214`.
  - Dark 3D blocks: Top `#2C2C34`, Front `#222228`, Side `#19191E`, Bottom `#141418`, Stroke `#454552`.

---

## 2. Typography & Scale Hierarchy (`1920×1080` Canvas)

1. **Plane 1 Giant 3D World-Space Monument Type (`z-index: 2`, Behind 3D Objects):**
   - `Inter Black 900` (`192px`, tracking `-9px`, line-height `0.84`) + `Space Grotesk Bold 700` (`216px`, tracking `-10px`).
   - Projected into 3D world coordinates (`z = -2.2`) behind the 3D model and tilted with the camera's perspective (`rotateX`, `rotateY`, `rotateZ`) so foreground 3D weights physically pass **in front of** the giant letters.
2. **Plane 3 Foreground Kinetic Headlines (`z-index: 4`, Top-Left Spatial Zone):**
   - `Inter Black 900` (`68px–76px`, tracking `-3.4px`, line-height `0.94`).
3. **Hero Numbers & Stat Lockups (`z-index: 4`, Top-Right Spatial Zone):**
   - `Space Grotesk Bold 700` (`64px–74px`, `tabular-nums`, tracking `-3.5px`).
4. **3D Tracked Callout Cards (`z-index: 3`, Left/Right Rails `x = 80` & `x = 1404`):**
   - Title: `Inter Black 900` (`18px`). Subtitle: `Inter Bold 700` (`14px`). Badge pill: `JetBrains Mono Bold 700` (`11px`).
5. **Single-Word Spoken Subtitle Rail (`y = 874`, `z-index: 4`):**
   - `Inter Black 900` (`30px`, tracking `-0.6px`), locked in fixed 9-word non-scrolling pages. Only the **single word currently being spoken** turns `#FF5A5F` with a `4px` `#FF5A5F` underline.

---

## 3. Multi-Plane 3D Depth & Continuous Camera Choreography

Every video must be an **active 3D camera reel**, never a static split-screen diagram:

- **3 Depth Planes:**
  - **Plane 1 (`z-index: 2`):** Giant 3D World-Space Monument Typography (`WorldBackdropTypography`).
  - **Plane 2 (`z-index: 3`):** Full-screen `THREE.PerspectiveCamera` 3D Stage (`Iso3DStage`), clipped strictly to `y = 92..860` so 3D geometry never bleeds into the top masthead or bottom subtitle rail.
  - **Plane 3 (`z-index: 4`):** Foreground Apple-style kinetic stat modules, live 3D camera telemetry (`3D CAM [x, y, z] · FOV`), single-word spoken caption rail, and hardware meter.
- **Mandatory 3D Camera Moves (`getCameraRig` in `src/Iso3D.tsx`):**
  1. **Crane Sweep + Dolly Zoom:** Low-angle orbital sweep (`fov 46° → 42°`) as model layers separate vertically to expose internal weight matrices.
  2. **Lateral Whip-Pan:** Fast spring-driven horizontal camera pan between contrasting 3D rigs (`pos.x: -3.2 → +3.2`) with banking roll (`±0.045 rad`).
  3. **Apple Silicon-Style 3D Weight Cutaway:** Outer casing peels open while the camera orbits into the gap between upper and lower internal subsystems (e.g., 4-layer Attention Crown vs. 3×3×3 exploded 27-voxel MLP lattice).
  4. **180° Orbital Surgical Assembly:** Camera orbits `-0.85 rad → +0.90 rad` while exploded 3D tensors compress 4× into a dense core and satellite modules dock via optical bridges.
  5. **First-Person 3D Tunnel Fly-Through:** Camera flies straight along `-Z` (`z: +8.4 → -0.6`, `fov = 50°`) through architectural portals/gates that open and light up as the narrator names each test.
  6. **Low-Angle Skyscraper Rise & Hero Pullback:** Camera cranes upward alongside extruding 3D benchmark towers, then pulls back out of the core for the finale.

---

## 4. Animation Physics & Word-Locked Timing

- **Zero Linear Transitions:** Every movement uses Remotion `spring()`:
  - **Snappy Spring:** `{ mass: 0.5, stiffness: 120, damping: 14 }`
  - **Bounce / Overshoot Spring:** `{ mass: 0.5, stiffness: 100, damping: 12 }`
  - **Smooth Camera Spring:** `{ mass: 0.9, stiffness: 70, damping: 18 }`
- **Sequential 3-Frame Stagger (`+3f`):** Kicker (`+3f`), Headline (`+6f`), Sub-block (`+9f`).
- **Spoken-Keyword Visual Triggers (`timeline.json` `triggers`):**
  - Stat cards, 3D cutaways, hydraulic crushes, logic gates, and benchmark columns **must never use arbitrary hardcoded frame delays**. They must trigger on the **exact local frame** (`tr.<keyword>`) when the narrator speaks that word.

---

## 5. Natural Human Audio & Word-Sync Law

1. **Conversational Neural Voice at Natural Speed:**
   - Use `en-US-AndrewMultilingualNeural` at `rate="-2%"` via `edge-tts` (`capture_subtitles=True` for `WordBoundary` events).
2. **Forbidden: `ffmpeg atempo` Time-Stretching:**
   - **NEVER** speed up or time-stretch TTS voiceover with `atempo`. Let the natural spoken duration of each scene (`~2.1 words/sec`, ~126 WPM) + `0.25s` lead-in + `0.65s` breathing pause determine `startFrame` and `endFrame` in `src/timeline.json`.
3. **Foley & Loudness Mastering:**
   - Layer subtle procedural `sfx` cues (`whoosh`, `pop`, `tick`, `sub_impact`, `braam`, `success`) locked to the exact `globalTriggers` frames in `timeline.json`, mixed at `-20 LUFS` under the voiceover and mastered to `-14 LUFS` stereo `48 kHz` (`public/music.wav`).

---

## 6. Five Things Our Motion Must Never Do

1. **Never use gradients, glows, bloom blowouts, or drop shadows.**
2. **Never use linear easing (`transition: all 0.5s linear`) or robotic constant-speed slides.**
3. **Never build a static split-screen slide deck when the camera should be flying through 3D space.**
4. **Never let 3D callout cards overlap each other or bleed into the top masthead (`y < 92`) or bottom subtitle rail (`y > 860`).**
5. **Never rush, warp, or `atempo`-stretch the voiceover, or allow visuals/subtitles to drift out of sync with the spoken word.**

---

## 7. Mandatory Visual QA Loop (`stills.mjs`)

Before rendering any full MP4, render an 8-tile contact sheet across the climax frame of every scene:
```bash
node stills.mjs 310 680 1050 1480 1880 2350 2700 2920
```
Inspect `out/contact.png` to verify:
- Giant 3D background typography is partially occluded *behind* the 3D model weights.
- Camera framing keeps the 3D subject centered within `y = 92..860` without clipping or grazing portal pillars.
- Left and right tracked leader-line callout cards have zero vertical overlap.
- The single-word `#FF5A5F` subtitle highlight matches the exact visual beat on screen.
