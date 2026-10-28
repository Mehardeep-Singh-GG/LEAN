# Workspace Engineering & Motion Rules (Anti-AI-Slop Standard)

Before designing, generating, or animating anything, read [`MOTION.md`](./MOTION.md) and [`brand.md`](./brand.md) in full.

Every colour, font, 3D camera rig, spring timing, and audio synchronization value comes from [`MOTION.md`](./MOTION.md).

The file sets the look, not the ambition. When I say go all out, go all out.

If something I ask for is not covered there, ask me rather than choosing for yourself.

When you have finished, check your own frames against [`MOTION.md`](./MOTION.md) using `node stills.mjs`, fix what fails, and only then show me.

---

## Core Engineering & Results Presentation Rules

1. **Zero Invented Numbers:** Always use exact, hardware-verified numbers from our experiments (`15.23 GB → 3.84 GiB`, `88.0% vs 12.0%` on 7B, `94.0% ± 0.0%` across 5 seeds on 3B, `75.0%` pure deductive logic with zero search, `Cliff's δ = 1.000`, `14–22 words/sec` in RAM, `4.08 GB` free RAM headroom, `18 KB` Pocket Index with `0.2 ms` lookup).
2. **Intuition First, Jargon Zero (Unless Requested):** In explainers and hackathon pitches, translate technical concepts into clear physical metaphors ("20% Thinking Engine vs. 80% Trivia Warehouse", "blunt 2-bit hammer vs. surgical 3-step compression") while preserving the exact empirical numbers.
3. **3-Plane 3D Camera Reels Over Static Slides:** Every motion video must use the 3-plane depth architecture in [`hackathon-video/src/Iso3D.tsx`](./hackathon-video/src/Iso3D.tsx) and [`hackathon-video/src/Video.tsx`](./hackathon-video/src/Video.tsx):
   - **Plane 1 (`z-index: 2`):** Giant 3D world-space monument typography positioned **behind** the 3D objects (`WorldBackdropTypography`).
   - **Plane 2 (`z-index: 3`):** Continuous `THREE.PerspectiveCamera` fly-through, orbital cutaway, opening up internal model weights, and 3D tracked leader-line callouts (`Iso3DStage`), clipped to `y = 92..860`.
   - **Plane 3 (`z-index: 4`):** Foreground Apple-style kinetic stat modules, live 3D camera telemetry, single-word `#FF5A5F` spoken subtitle highlight, and hardware RAM meter.
4. **Natural Human Audio Sync (`build_audio.py`):** Always use `en-US-AndrewMultilingualNeural` at `rate="-2%"` with **zero `ffmpeg atempo` stretching**. Let natural speech duration drive `timeline.json` scene bounds, and trigger every visual beat on its exact spoken `WordBoundary` frame (`triggers`).
