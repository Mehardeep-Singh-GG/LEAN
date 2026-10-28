---
trigger: always_on
---

# Anti-AI-Slop 3D Motion & Engineering Results Rules

Before designing, generating, or animating any video, reel, chart, or visual artifact, you **MUST** read [`/Users/mac/llm/MOTION.md`](file:///Users/mac/llm/MOTION.md) and [`/Users/mac/llm/brand.md`](file:///Users/mac/llm/brand.md) in full, and use the reusable [`anti-slop-3d-reel`](file:///Users/mac/llm/.agents/skills/anti-slop-3d-reel/SKILL.md) skill.

## 1. How All Results & Explanations Must Be Delivered
- **Hardware-Verified Precision:** Never invent, approximate, or omit real experimental results. Always cite exact measured figures (`15.23 GB → 3.84 GiB`, `88.0% vs 12.0%` on 7B, `94.0% ± 0.0%` across 5 seeds on 3B, `75.0%` standalone pure deductive logic with zero search, `Cliff's δ = 1.000`, `14–22 words/sec` in RAM, `4.08 GB` free RAM headroom, `18 KB` Pocket Index at `0.2 ms`).
- **Physical Intuition Over Buzzwords:** Explain mechanisms using tangible architectural metaphors ("20% Thinking Engine vs. 80% Trivia Warehouse", "opening up the model weights") backed by exact numbers.

## 2. Mandatory Anti-AI-Slop 3D Motion Architecture
Never produce generic "AI slop" (flat linear CSS transitions, neon purple/blue gradients, glowing drop shadows, simultaneous element pops, or static side-by-side slide decks). Every video must follow these 5 laws:

1. **Strict Matte Palette (`#121214`, `#FF5A5F`, `#EAEAEA`):**
   - Background: `#121214` (flat matte obsidian).
   - Accents: `#FF5A5F` (Signal Coral) and `#EAEAEA` (Off-White) plus flat directional 3D face shading and `#26262B` structural rules.
   - **Zero** linear/radial gradients, **zero** neon glows, **zero** drop shadows.
2. **3-Plane Depth & Giant Text *Behind* 3D Objects:**
   - **Plane 1 (`z-index: 2`):** Giant 3D world-space monument typography (`WorldBackdropTypography`) projected behind the 3D scene (`z = -2.2`) with 3D perspective tilt (`rotateX`, `rotateY`, `rotateZ`) so foreground 3D model weights physically pass **in front of** the giant letters.
   - **Plane 2 (`z-index: 3`):** Continuous `THREE.PerspectiveCamera` (`fov`, `pos`, `target`, banking `roll`) that cranes, whip-pans, dives into opened 3D model weights (exposing internal Q/K/V weight matrices and 3×3×3 exploded MLP voxel lattices), and flies straight through 3D portals (`Iso3DStage`), clipped strictly to `y = 92..860`.
   - **Plane 3 (`z-index: 4`):** Apple-style floating kinetic stat modules, live 3D camera telemetry (`3D CAM [x, y, z] · FOV`), and 3D-tracked leader-line callout cards (`o───`) with non-overlapping left/right vertical slots.
3. **Snappy Spring Physics & 3-Frame Stagger (`+3f`):**
   - Never use `linear` easing. Use Remotion `spring({ frame, fps, config: { mass: 0.5, stiffness: 120, damping: 14 } })` or `{ damping: 12, mass: 0.5, stiffness: 100 }`.
4. **Natural Human Voiceover & Word-Locked Sync (`build_audio.py`):**
   - Use `en-US-AndrewMultilingualNeural` at `rate="-2%"` (`~2.1 words/sec`).
   - **NEVER** apply `ffmpeg atempo` time-stretching to voiceover audio. Let natural speech duration + `0.65s` scene pauses set the exact frame bounds in `src/timeline.json`.
   - Trigger every stat card and 3D animation on its exact spoken `WordBoundary` frame (`triggers`), and highlight strictly **one word at a time** in `#FF5A5F` on the bottom 9-word grid rail.
5. **Mandatory Visual QA (`stills.mjs`):**
   - Always render an 8-scene contact sheet (`node stills.mjs ...`) and inspect `out/contact.png` to verify depth occlusion, camera framing, and zero card overlaps before rendering the final MP4.
