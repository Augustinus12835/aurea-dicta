# Aurea Dicta

**Latin:** "Golden Words"

AI pipeline that transforms lecture recordings into short, focused concept videos.

A 2-3 hour lecture becomes 8-10 videos (5-8 min each) with AI narration, illustrated frames, math animations, and subtitles.

## Pipeline

```
Lecture (YouTube URL or local .mp4)
  -> Transcribe -> Segment into concepts
  -> Per-video: Brief -> Charts -> Script -> Verify Math -> Frames -> TTS -> Animate -> Compile
  -> 8-10 concept videos
```

**Key features:**
- YouTube URL or local video input
- Claude for scripts, math verification, and Manim animation code
- Gemini for illustrated frames
- Manim for animated math walkthroughs (auto-selected for frames >60s)
- ElevenLabs TTS with voice cloning
- Fully resumable (file-based state detection)

## Quick Start

```bash
git clone https://github.com/Augustinus12835/aurea-dicta.git
cd aurea-dicta
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` with your API keys:

| Key | Service | Required |
|-----|---------|----------|
| `ANTHROPIC_API_KEY` | Claude (scripts, math, animation) | Yes |
| `GOOGLE_CLOUD_API_KEY` | Gemini (frame images) | Yes |
| `ELEVENLABS_API_KEY` | TTS | Yes |
| `ELEVENLABS_VOICE_ID` | Voice ID | Yes |
| `ASSEMBLYAI_API_KEY` | Transcription (local files only) | Optional |

Run:

```bash
# From YouTube
python scripts/pipeline.py run "https://www.youtube.com/watch?v=VIDEO_ID" --no-review

# From local video
python scripts/pipeline.py run MY_LECTURE --no-review

# Single video, resume from a step
python scripts/pipeline.py video MY_LECTURE/Video-3 --from script

# Check status
python scripts/pipeline.py status MY_LECTURE
```

## Output Structure

```
pipeline/MY_LECTURE/
├── transcript.json
├── segments.json
└── Video-1/
    ├── video_brief.md
    ├── script.md
    ├── math_verification.json    # Math steps + TTS-friendly narration
    ├── frames/
    │   ├── frame_0.png           # Gemini illustrations
    │   ├── frame_2.mp4           # Manim animations (long math frames)
    │   └── frame_2_manim.py      # Generated Manim source
    ├── audio/
    ├── final_video.mp4
    └── subtitles.srt
```

## Math Animation

For math-heavy videos, frames longer than 60 seconds are automatically animated with Manim instead of static images. The system:

1. Detects qualifying frames by audio duration
2. Auto-classifies layout (full whiteboard, split-screen with graph, or number line)
3. Generates Manim code via Claude with word-level timing from Whisper
4. Renders 1080p30 animations synced to narration

Skipped entirely for non-math content.

```bash
# Manual animation of a specific frame
python scripts/generate_math_animation.py pipeline/MY_LECTURE/Video-1 --frame 5

# Force regeneration
python scripts/generate_math_animation.py pipeline/MY_LECTURE/Video-1 --force
```

## Fixing Frames

```bash
# Regenerate a specific Gemini frame with instructions
python scripts/regenerate_frame.py pipeline/MY_LECTURE/Video-1 5 \
    --instruction "Arrow should point downward"

# Recompile after fixing frames
python scripts/compile_video.py pipeline/MY_LECTURE/Video-1
```

## Estimated Costs

| Service | Per 3-hour lecture | Cost |
|---------|-------------------|------|
| Claude | ~70K tokens | ~$2.50 |
| Gemini | ~100 images | ~$3.00 |
| ElevenLabs | ~120K chars | ~$1.00 |
| AssemblyAI | ~3 hours | ~$0.80 (free with YouTube) |
| **Total** | | **~$6.50-7.30** |

## License

MIT License - see LICENSE file.
