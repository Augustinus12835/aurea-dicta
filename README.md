# Aurea Dicta

**Latin:** "Golden Words"

Automated pipeline for converting long lecture recordings into short, engaging concept videos.

## Overview

Aurea Dicta transforms 2-3 hour lecture recordings into 8-10 focused concept videos (5-8 minutes each), complete with:
- AI-generated narration scripts
- Data visualizations (matplotlib) and conceptual diagrams (Gemini)
- Professional TTS narration with cloned voice
- Whisper-aligned subtitles

## Features

- **YouTube Support:** Process any YouTube video directly via URL (no download needed)
- **Automatic Transcription:** YouTube captions or AssemblyAI for local videos
- **Intelligent Segmentation:** Claude Opus 4.5 identifies logical concept breaks
- **Script Generation:** AI writes concise, educational narration scripts
- **Data Charts:** Matplotlib generates accurate data visualizations
- **Conceptual Slides:** Gemini creates hand-drawn style diagrams
- **Voice Cloning:** ElevenLabs TTS with custom voice support
- **Video Compilation:** FFmpeg with Whisper-aligned subtitles
- **Resumable Pipeline:** File-based state detection - quit anytime, resume later

## Directory Structure

```
aurea_dicta/
├── inputs/                    # Place your lecture videos here (.mp4)
├── pipeline/                  # Generated output (created automatically)
├── scripts/                   # Pipeline scripts
│   ├── pipeline.py            # Main CLI tool
│   ├── utils/                 # API clients
│   │   ├── claude_client.py
│   │   └── gemini_client.py
│   ├── transcribe_lecture.py
│   ├── clean_transcript.py
│   ├── segment_concepts.py
│   ├── generate_briefs.py
│   ├── generate_data_charts.py
│   ├── generate_scripts.py
│   ├── generate_slides_gemini.py
│   ├── generate_tts_elevenlabs.py
│   ├── compile_video.py
│   └── regenerate_frame.py
├── templates/                 # Style guides and templates
│   ├── teaching_style_guide.md
│   ├── slide_style_guide.md
│   ├── video_brief.md
│   ├── script.md
│   └── plan.md
├── .env.example               # Template for API keys
└── requirements.txt
```

## Quick Start

### 1. Prerequisites

- Python 3.9+
- FFmpeg (for video processing)
- API keys for all services (see below)

**Install FFmpeg:**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 2. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/aurea_dicta.git
cd aurea_dicta

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` with your actual API keys:

| Key | Service | Required | Get from |
|-----|---------|----------|----------|
| `ANTHROPIC_API_KEY` | Claude (script generation) | Yes | [console.anthropic.com](https://console.anthropic.com/) |
| `GOOGLE_CLOUD_API_KEY` | Gemini (image generation) | Yes | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `ELEVENLABS_API_KEY` | TTS voice synthesis | Yes | [elevenlabs.io](https://elevenlabs.io/) |
| `ELEVENLABS_VOICE_ID` | Your voice ID | Yes | ElevenLabs dashboard |
| `ASSEMBLYAI_API_KEY` | Transcription | Optional* | [assemblyai.com](https://www.assemblyai.com/) |
| `FRED_API_KEY` | Economic data charts | Optional | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `ALPHA_VANTAGE_API_KEY` | Financial data | Optional | [alphavantage.co](https://www.alphavantage.co/support/#api-key) |

*AssemblyAI is only required for local video files. YouTube videos use YouTube's built-in captions.

### 4. Add Your Source

**Option A: YouTube URL (recommended for online content)**
```bash
# No download needed - just use the URL directly
python scripts/pipeline.py run "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Option B: Local Video File**
```bash
cp /path/to/your/lecture.mp4 inputs/MY_LECTURE.mp4
python scripts/pipeline.py run MY_LECTURE
```

### 5. Run the Pipeline

```bash
# Activate virtual environment
source venv/bin/activate

# From YouTube URL (auto-creates folder from video title)
python scripts/pipeline.py run "https://www.youtube.com/watch?v=7xTGNNLPyMI"

# From local video
python scripts/pipeline.py run MY_LECTURE
```

## Pipeline Commands

### Run Full Pipeline

```bash
# From YouTube URL
python scripts/pipeline.py run "https://www.youtube.com/watch?v=VIDEO_ID"

# From local video
python scripts/pipeline.py run MY_LECTURE

# Process only a specific video segment
python scripts/pipeline.py run MY_LECTURE --video 3

# Skip review checkpoints (batch mode)
python scripts/pipeline.py run MY_LECTURE --no-review

# Force restart from a specific step
python scripts/pipeline.py run MY_LECTURE --from segment
```

### Run Single Video

```bash
# Process just one video (after week-level steps are done)
python scripts/pipeline.py video MY_LECTURE/Video-3

# Resume from a specific step
python scripts/pipeline.py video MY_LECTURE/Video-3 --from script
```

### Check Status

```bash
# Show status for entire lecture
python scripts/pipeline.py status MY_LECTURE

# Show status for specific video
python scripts/pipeline.py status MY_LECTURE/Video-3
```

## Pipeline Flow

```
    YouTube URL                Local Video
         │                          │
         ▼                          ▼
  (YouTube captions)         (AssemblyAI)
         │                          │
         └──────────┬───────────────┘
                    ▼
┌─────────────────────────────────┐
│ WEEK-LEVEL STEPS                │
│ 1. Transcribe (auto-detected)   │
│ 2. Clean transcript             │
│ 3. Segment into concepts        │
│    >>> REVIEW CHECKPOINT        │
└─────────────────────────────────┘
         │
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
┌─────────────────────────────────┐
│ PER-VIDEO STEPS (for each)      │
│ 4. Generate video brief         │
│    >>> REVIEW CHECKPOINT        │
│ 5. Generate data charts         │
│    >>> REVIEW CHECKPOINT        │
│ 6. Generate narration script    │
│    >>> REVIEW CHECKPOINT        │
│ 7. Generate frame images        │
│    >>> REVIEW CHECKPOINT        │
│ 8. Generate TTS audio           │
│    >>> REVIEW CHECKPOINT        │
│ 9. Compile final video          │
└─────────────────────────────────┘
         │
         ▼
   8-10 Concept Videos
```

### Review Checkpoints

At each checkpoint, you can:
- `[c]` Continue to next step
- `[q]` Quit (resume later with same command)

The pipeline is fully resumable - it detects state from output files. Edit files externally and resume; changes are respected.

## Output Structure

After processing, your `pipeline/` folder will contain:

```
pipeline/MY_LECTURE/
├── source.json             # Source metadata (YouTube URL/video ID)
├── transcript.json         # Raw transcription
├── content_cleaned.txt     # Cleaned content
├── segments.json           # Segmentation data
├── plan.md                 # Human-readable plan
└── Video-1/                # Each video segment
    ├── content.txt         # Segment content
    ├── video_brief.md      # Teaching structure
    ├── visual_specs.json   # Visual specifications
    ├── diagrams/           # Data charts (matplotlib)
    │   ├── visual_1.png
    │   └── visual_1_code.py
    ├── script.md           # Frame-by-frame narration
    ├── frames/             # Slide images (Gemini)
    │   ├── frame_0.png
    │   ├── frame_1.png
    │   └── ...
    ├── audio/              # TTS audio files
    │   ├── frame_0.mp3
    │   └── ...
    ├── final_video.mp4     # Compiled video
    └── subtitles.srt       # Aligned subtitles
```

## Frame Generation

Frame generation is resumable - it only generates missing frames by default:

```bash
# Generate frames (skips existing ones)
python scripts/generate_slides_gemini.py pipeline/MY_LECTURE/Video-1

# Force regenerate all frames
python scripts/generate_slides_gemini.py pipeline/MY_LECTURE/Video-1 --force

# Generate a specific frame only
python scripts/generate_slides_gemini.py pipeline/MY_LECTURE/Video-1 --frame 5

# Continue on errors (don't stop at first failure)
python scripts/generate_slides_gemini.py pipeline/MY_LECTURE/Video-1 --continue-on-error

# Verbose mode (see full prompts)
python scripts/generate_slides_gemini.py pipeline/MY_LECTURE/Video-1 --verbose
```

### Regenerating Specific Frames

After reviewing generated frames, regenerate specific ones with corrections:

```bash
# Basic regeneration
python scripts/regenerate_frame.py pipeline/MY_LECTURE/Video-1 5

# With correction instruction
python scripts/regenerate_frame.py pipeline/MY_LECTURE/Video-1 5 \
    --instruction "Make the arrow point downward, not upward"

# Verbose mode (see full prompt)
python scripts/regenerate_frame.py pipeline/MY_LECTURE/Video-1 5 -v
```

**Tips for better Gemini corrections:**
- Be specific about shapes: "diagonal line sloping DOWNWARD from upper-left"
- Describe exact positions: "label on the RIGHT side, not left"
- For complex technical diagrams, consider matplotlib generation instead

## Running Individual Scripts

For debugging or custom workflows, run scripts individually:

```bash
# 1. Transcribe
python scripts/transcribe_lecture.py inputs/MY_LECTURE.mp4

# 2. Clean transcript
python scripts/clean_transcript.py pipeline/MY_LECTURE

# 3. Segment into concepts
python scripts/segment_concepts.py pipeline/MY_LECTURE

# 4. Generate briefs
python scripts/generate_briefs.py pipeline/MY_LECTURE --video 1

# 5. Generate data charts
python scripts/generate_data_charts.py pipeline/MY_LECTURE --video 1

# 6. Generate scripts
python scripts/generate_scripts.py pipeline/MY_LECTURE --video 1

# 7. Generate slides
python scripts/generate_slides_gemini.py pipeline/MY_LECTURE/Video-1

# 8. Generate TTS audio
python scripts/generate_tts_elevenlabs.py pipeline/MY_LECTURE/Video-1/script.md

# 9. Compile video
python scripts/compile_video.py pipeline/MY_LECTURE/Video-1
```

## Troubleshooting

### Common Issues

**"ELEVENLABS_VOICE_ID not found"**
- Add `ELEVENLABS_VOICE_ID=your_voice_id` to your `.env` file
- Find your voice ID in the ElevenLabs dashboard

**"FFmpeg not found"**
- Install FFmpeg and ensure it's in your PATH
- Test with: `ffmpeg -version`

**Gemini generates incorrect diagrams**
1. Use `regenerate_frame.py` with detailed `--instruction`
2. Be very specific about visual elements
3. For technical charts (graphs, payoff diagrams), consider using matplotlib instead

**Gemini frame generation fails ("No image in response")**
- Run with `--verbose` to see detailed error info:
  ```bash
  python scripts/generate_slides_gemini.py pipeline/MY_LECTURE/Video-1 --frame 5 --verbose
  ```
- Common causes:
  - **Safety filters**: Content may trigger content policy (religious, medical topics)
  - **Rate limiting**: Too many requests - wait and retry
  - **Complex prompts**: Simplify the visual description in `visual_specs.json`
- The error message shows `finishReason` and blocked categories for diagnosis
- Use `--continue-on-error` to generate other frames despite failures

**TTS reads visual annotations**
- The `[Visual: ...]` annotations are automatically stripped from narration
- If you see this issue, ensure you're using the latest scripts

**Pipeline stuck or errors**
- Check `python scripts/pipeline.py status MY_LECTURE` for current state
- Delete the problematic output file to regenerate it
- Resume with the same command

**Charts not generating**
- Ensure `visual_specs.json` has `type: "data_chart"` entries
- Check that required API keys (FRED, Alpha Vantage) are set for data sources
- Run with `--force` to regenerate existing charts

### Debugging Tips

1. **Check pipeline status first:**
   ```bash
   python scripts/pipeline.py status MY_LECTURE
   ```

2. **Run individual scripts** to isolate issues:
   ```bash
   python scripts/generate_slides_gemini.py pipeline/MY_LECTURE/Video-1
   ```

3. **Delete output to regenerate:**
   ```bash
   rm pipeline/MY_LECTURE/Video-1/script.md
   python scripts/pipeline.py video MY_LECTURE/Video-1 --from script
   ```

4. **Check generated code files:**
   - Chart code is saved to `diagrams/*_code.py`
   - Run manually to debug: `python pipeline/MY_LECTURE/Video-1/diagrams/visual_1_code.py`

## Estimated Costs

| Service | Usage per 3-hour lecture | Cost |
|---------|--------------------------|------|
| AssemblyAI | ~3 hours | ~$0.80 (free with YouTube) |
| Claude Opus 4.5 | ~70K tokens | ~$2.50 |
| Gemini Image | ~100 images | ~$3.00 |
| ElevenLabs TTS | ~120K characters | ~$1.00 |
| **Total (local video)** | | **~$7.30** |
| **Total (YouTube)** | | **~$6.50** |

## Templates

Customize the output style by editing templates in `templates/`:

| Template | Purpose |
|----------|---------|
| `teaching_style_guide.md` | Narration voice and tone |
| `slide_style_guide.md` | Visual style specifications |
| `video_brief.md` | Video planning structure |
| `script.md` | Narration script format |
| `plan.md` | Lecture segmentation format |

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [AssemblyAI](https://www.assemblyai.com/) for transcription
- [Anthropic Claude](https://www.anthropic.com/) for content generation
- [Google Gemini](https://deepmind.google/technologies/gemini/) for image generation
- [ElevenLabs](https://elevenlabs.io/) for voice synthesis
- [FFmpeg](https://ffmpeg.org/) for video processing
