# Aurea Dicta Pipeline Scripts

This directory contains the automated pipeline scripts for converting lecture videos into concept videos.

## Pipeline Scripts

### Core Pipeline (in order)

1. **transcribe_lecture.py** - Transcribe lecture using Whisper
   ```bash
   python scripts/transcribe_lecture.py inputs/YOUR_LECTURE.mp4
   ```
   Output: `pipeline/<lecture_id>/transcript.json`

2. **segment_concepts.py** - Segment lecture into concept videos using Claude
   ```bash
   python scripts/segment_concepts.py pipeline/YOUR_LECTURE
   ```
   Output: `pipeline/<lecture_id>/segments.json`, `plan.md`

3. **generate_briefs.py** - Generate video briefs using Claude
   ```bash
   python scripts/generate_briefs.py pipeline/YOUR_LECTURE
   ```
   Output: `pipeline/<lecture_id>/Video-N/video_brief.md`

4. **generate_scripts.py** - Generate narration scripts using Claude
   ```bash
   python scripts/generate_scripts.py pipeline/YOUR_LECTURE
   ```
   Output: `pipeline/<lecture_id>/Video-N/script.md`

5. **generate_slides.py** - Generate slide images using Gemini
   ```bash
   python scripts/generate_slides.py pipeline/YOUR_LECTURE
   ```
   Output: `pipeline/<lecture_id>/Video-N/frames/*.png`

6. **generate_tts_elevenlabs.py** - Generate TTS audio using ElevenLabs
   ```bash
   python scripts/generate_tts_elevenlabs.py pipeline/YOUR_LECTURE/Video-1/script.md
   ```
   Output: `pipeline/<lecture_id>/Video-N/audio/*.mp3`

7. **compile_video.py** - Compile final video with subtitles
   ```bash
   python scripts/compile_video.py pipeline/YOUR_LECTURE/Video-1
   ```
   Output: `pipeline/<lecture_id>/Video-N/final_video.mp4`

### Pipeline Control

**pipeline_control.py** - Orchestrate full pipeline with pause/resume
```bash
# Run full pipeline
python scripts/pipeline_control.py run inputs/YOUR_LECTURE.mp4

# Approve review points
python scripts/pipeline_control.py approve-plan YOUR_LECTURE
python scripts/pipeline_control.py approve-content YOUR_LECTURE

# Check status
python scripts/pipeline_control.py status YOUR_LECTURE

# Resume from specific step
python scripts/pipeline_control.py resume YOUR_LECTURE --from scripts
```

### Utility Scripts

- **generate_data_charts.py** - Generate matplotlib charts from data specifications
- **generate_images_gemini.py** - Generate images using Gemini API

### API Clients (utils/)

- **claude_client.py** - Anthropic Claude API wrapper
- **gemini_client.py** - Google Gemini API wrapper

## Review Points

The pipeline pauses at two review points:

1. **After Segmentation** - Review `plan.md` before generating briefs
2. **After Scripts/Slides** - Review scripts and slides before TTS

## Environment Variables

Required in `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
ELEVENLABS_API_KEY=...
google_cloud_api_key=...
```

Optional:
```
fred_api_key=...  # For FRED economic data charts
```

---

_Last Updated: 2025-12-06_
_Project: Aurea Dicta - Automated Lecture-to-Video Pipeline_
