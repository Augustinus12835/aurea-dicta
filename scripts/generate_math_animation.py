#!/usr/bin/env python3
"""
Math Animation Generator for Aurea Dicta

Generates Manim-animated video clips for math-heavy frames, replacing static
Gemini PNG images with step-by-step animated walkthroughs synced to narration.

Qualifying frames must have:
- math_verification.json with requires_verification: true
- Non-empty math_steps
- verification_status in ("correct", "corrected")

Uses full-screen whiteboard layout for animations.

Usage:
    python scripts/generate_math_animation.py pipeline/LECTURE/Video-N
    python scripts/generate_math_animation.py pipeline/LECTURE/Video-N --frame 2
    python scripts/generate_math_animation.py pipeline/LECTURE/Video-N --force
"""

import os
import sys
import json
import re
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add parent to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.claude_client import ClaudeClient
from scripts.utils.script_parser import load_script


def get_audio_duration_ffprobe(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def get_video_info(video_path: str) -> Dict:
    """Get video resolution and duration using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-show_entries', 'format=duration',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    info = {}
    if data.get('streams'):
        info['width'] = data['streams'][0].get('width')
        info['height'] = data['streams'][0].get('height')
        info['fps'] = data['streams'][0].get('r_frame_rate')
    if data.get('format'):
        info['duration'] = float(data['format'].get('duration', 0))
    return info


def load_math_verification(video_folder: str) -> Optional[Dict]:
    """Load math_verification.json if it exists."""
    path = os.path.join(video_folder, "math_verification.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_system_prompt() -> str:
    """Load the Manim system prompt template."""
    prompt_path = Path(__file__).parent.parent / "templates" / "manim_system_prompt.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_qualifying_frames(video_folder: str, specific_frame: Optional[int] = None) -> List[int]:
    """
    Identify frames that qualify for Manim animation.

    A frame qualifies if it has:
    - requires_verification: true in math_verification.json
    - Non-empty math_steps list
    - verification_status in ("correct", "corrected")
    """
    math_data = load_math_verification(video_folder)
    if not math_data:
        return []

    qualifying = []
    for frame_key, frame_data in math_data.get("frames", {}).items():
        frame_num = int(frame_key)

        if specific_frame is not None and frame_num != specific_frame:
            continue

        if not frame_data.get("requires_verification"):
            continue
        if frame_data.get("verification_status") not in ("correct", "corrected"):
            continue
        if not frame_data.get("math_steps"):
            continue

        qualifying.append(frame_num)

    return sorted(qualifying)


ANIMATION_DURATION_THRESHOLD = 60  # seconds — primary selection knob
CONTAGION_DURATION_THRESHOLD = 25  # seconds — lower bar for adjacent frames
NEW_PROBLEM_PHRASES = [
    "consider", "new problem", "new example", "let's try a different",
    "here's another", "next example", "moving on"
]


def select_frames_for_animation(
    video_folder: str,
    math_data: dict,
    script_data
) -> Tuple[List[int], Dict[int, Dict]]:
    """
    Select which frames to animate based on audio duration and content.

    Primary signal: audio_duration >= 60s (long static image = retention drop)
    Secondary: cross-frame contagion for multi-frame examples

    Returns (selected_frame_numbers, frame_info_map).
    """
    audio_dir = os.path.join(video_folder, 'audio')
    frame_info_map = {}

    for frame_key, frame_data in math_data.get("frames", {}).items():
        frame_num = int(frame_key)
        if not frame_data.get("requires_verification"):
            continue
        if frame_data.get("verification_status") not in ("correct", "corrected"):
            continue
        if not frame_data.get("math_steps"):
            continue

        # Measure audio duration
        audio_path = os.path.join(audio_dir, f"frame_{frame_num}.mp3")
        if not os.path.exists(audio_path):
            continue
        duration = get_audio_duration_ffprobe(audio_path)

        # Skip title frames
        is_title = (frame_num == 0)
        if not is_title and script_data:
            script_frame = script_data.get_frame(frame_num)
            if script_frame and hasattr(script_frame, 'visual'):
                is_title = getattr(script_frame.visual, 'frame_type', '') == 'title'

        frame_info_map[frame_num] = {
            'duration': duration,
            'math_steps': len(frame_data.get("math_steps", [])),
            'narration': frame_data.get("natural_narration", ""),
            'is_title': is_title,
        }

    # Primary selection: long frames with math
    candidates = []
    for frame_num, info in frame_info_map.items():
        if not info['is_title'] and info['duration'] >= ANIMATION_DURATION_THRESHOLD:
            candidates.append(frame_num)

    # Cross-frame contagion: pull in adjacent frames that continue the same example
    animated_set = set(candidates)
    for frame_num in sorted(frame_info_map.keys()):
        if frame_num in animated_set:
            continue
        info = frame_info_map[frame_num]
        if info['is_title'] or info['duration'] < CONTAGION_DURATION_THRESHOLD:
            continue

        # Check if adjacent to an animated frame
        prev_animated = (frame_num - 1) in animated_set
        next_animated = (frame_num + 1) in animated_set

        if not (prev_animated or next_animated):
            continue

        # Check narration doesn't start a NEW problem (it's a continuation)
        narration_lower = info['narration'][:200].lower()
        starts_new = any(p in narration_lower for p in NEW_PROBLEM_PHRASES)

        if not starts_new:
            animated_set.add(frame_num)

    return sorted(animated_set), frame_info_map


def transcribe_with_whisper(audio_path: str) -> Dict:
    """
    Transcribe audio with Whisper to get word-level timestamps.

    Returns the raw Whisper result dict with segments and word timestamps.
    """
    import whisper
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(f"      Transcribing {os.path.basename(audio_path)} with Whisper...")
    model = whisper.load_model("small")
    result = model.transcribe(audio_path, word_timestamps=True, language="en")
    return result


def align_words_to_script(script_text: str, whisper_result: Dict) -> List[Dict]:
    """
    Align known script text (ground truth) to Whisper word timestamps.

    Uses Whisper's timing but replaces its transcription with the correct
    script text, avoiding misreadings of math terms.

    Returns list of {word, start, end} with corrected text.
    """
    # Extract raw Whisper words
    whisper_words = []
    for segment in whisper_result.get('segments', []):
        if 'words' in segment:
            for w in segment['words']:
                whisper_words.append({
                    'word': w.get('word', '').strip(),
                    'start': w['start'],
                    'end': w['end']
                })

    if not whisper_words:
        return []

    # Tokenize script text
    script_words = script_text.replace('\n', ' ').split()

    if not script_words:
        return []

    aligned = []

    if len(script_words) == len(whisper_words):
        # Direct 1-to-1 mapping — best case
        for sw, ww in zip(script_words, whisper_words):
            aligned.append({
                'word': sw,
                'start': round(ww['start'], 2),
                'end': round(ww['end'], 2)
            })
    else:
        # Word counts differ — proportional mapping
        total_duration = whisper_words[-1]['end'] - whisper_words[0]['start']
        time_per_word = total_duration / len(script_words)
        current_time = whisper_words[0]['start']

        for sw in script_words:
            aligned.append({
                'word': sw,
                'start': round(current_time, 2),
                'end': round(current_time + time_per_word, 2)
            })
            current_time += time_per_word

    return aligned


def format_word_transcript(aligned_words: List[Dict]) -> str:
    """
    Format aligned word timestamps into a compact transcript for Claude.

    Groups words into ~5-word chunks with the start time of the first word,
    keeping the prompt concise while giving Claude precise timing.
    """
    if not aligned_words:
        return "(no transcript available)"

    lines = []
    chunk_size = 5
    for i in range(0, len(aligned_words), chunk_size):
        chunk = aligned_words[i:i + chunk_size]
        time = chunk[0]['start']
        words = ' '.join(w['word'] for w in chunk)
        lines.append(f"[{time:6.2f}s] {words}")

    return '\n'.join(lines)


GRAPH_KEYWORDS = [
    "curve", "axes", "plot", "tangent", "shad", "region",
    "peak", "valley", "rising", "falling", "concav",
]
NUMBER_LINE_KEYWORDS = [
    "number line", "sign chart", "sign analysis", "test point",
]


def detect_layout(visual_desc: str, narration: str, math_steps: List[Dict]) -> str:
    """
    Detect which layout template to recommend based on frame content.

    Graph detection uses visual_desc only (describes what to draw).
    Number line detection uses visual_desc + narration.

    Returns 'A' (full whiteboard), 'B' (split screen), or 'C' (steps + number line).
    """
    visual_lower = visual_desc.lower()
    narration_lower = narration.lower()

    has_graph = any(kw in visual_lower for kw in GRAPH_KEYWORDS)
    has_number_line = any(
        kw in visual_lower or kw in narration_lower
        for kw in NUMBER_LINE_KEYWORDS
    )

    if has_graph and not has_number_line:
        return 'B'
    if has_number_line and not has_graph:
        return 'C'
    if has_graph and has_number_line:
        # Both — prefer split screen (graph is more spatially demanding)
        return 'B'
    return 'A'


LAYOUT_DESCRIPTIONS = {
    'A': "Layout A: Full Whiteboard — pure algebraic derivation, no graphs or number lines. Use the full-width add_step() helper.",
    'B': "Layout B: Split Screen — graph on LEFT (centered x=-3.3), steps on RIGHT (centered x=3.5). NEVER put graph above steps. FadeOut graph completely when no longer referenced.",
    'C': "Layout C: Steps Above + Number Line Below — algebraic steps in top zone (SCROLL_BOTTOM=-0.8), number line pinned at bottom (y=-2.5).",
}


def build_claude_prompt(
    narration: str,
    math_steps: List[Dict],
    visual_desc: str,
    total_duration: float,
    frame_number: int,
    word_transcript: str,
) -> str:
    """Build the user prompt for Claude to generate Manim code."""
    steps_text = ""
    for step in math_steps:
        steps_text += (
            f"step {step['step']}: "
            f"`{step['expression']}` — {step['operation']}"
        )
        if step.get('note'):
            steps_text += f" ({step['note']})"
        steps_text += "\n"

    layout = detect_layout(visual_desc, narration, math_steps)
    layout_desc = LAYOUT_DESCRIPTIONS[layout]

    return f"""Generate a Manim Scene class `MathAnimation` for this math frame.

FRAME NUMBER: {frame_number}
TOTAL DURATION: {total_duration:.1f} seconds (animation must fill this exactly)

LAYOUT_HINT: {layout_desc}

NARRATION (what the speaker says during this animation):
{narration}

VISUAL DESCRIPTION:
{visual_desc}

MATH STEPS (in order — YOU decide when each appears based on the transcript):
{steps_text}

WORD-LEVEL TRANSCRIPT (ground-truth text with precise timestamps):
{word_transcript}

REQUIREMENTS:
1. The Scene class must be named `MathAnimation`
2. Total animation duration must be {total_duration:.1f}s (sum of all run_time + wait calls)
3. **Follow the LAYOUT_HINT above.** Use the matching layout template from the system prompt.
4. Read the word-level transcript carefully. Each math step's animation should BEGIN when the narrator starts introducing that concept — find the words in the transcript that correspond to each step
5. NEVER have dead time with just a title card — start showing math content within the first 1-2 seconds
6. Use the color scheme from the system prompt (dark bg, blue math, orange highlights, green answer)
7. The final step/answer should remain visible until the end
8. Use the `add_step()` whiteboard helper from the system prompt for sequential derivation steps
9. Return ONLY the Python code, no explanation

Return the complete Python code starting with `from manim import *`."""


def generate_manim_code(
    claude: ClaudeClient,
    system_prompt: str,
    user_prompt: str
) -> str:
    """Call Claude to generate Manim Scene code."""
    response = claude.generate(
        prompt=user_prompt,
        system=system_prompt,
        max_tokens=32000,
        temperature=0.3,
        thinking=True,
        effort="medium"
    )

    # Extract code block if wrapped in markdown fences
    code = response.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    return code


def render_manim_scene(
    scene_code: str,
    output_path: str,
    total_duration: float
) -> Tuple[bool, str]:
    """
    Write Manim code to a temp file, render it, and copy output.

    Returns (success, message).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy any referenced images into the temp dir so Manim can find them.
        # Scan for ImageMobject("...") paths and rewrite to local filenames.
        import re as _re
        for match in _re.finditer(r'ImageMobject\(["\'](.+?)["\']\)', scene_code):
            img_path = match.group(1)
            if os.path.isabs(img_path) and os.path.exists(img_path):
                local_name = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(tmpdir, local_name))
                scene_code = scene_code.replace(img_path, local_name)

        scene_file = os.path.join(tmpdir, "scene.py")
        with open(scene_file, 'w', encoding='utf-8') as f:
            f.write(scene_code)

        # Render with Manim
        cmd = [
            sys.executable, '-m', 'manim', 'render',
            '-r', '1920,1080',
            '--fps', '30',
            '--format', 'mp4',
            '-o', 'output.mp4',
            '--media_dir', os.path.join(tmpdir, 'media'),
            scene_file,
            'MathAnimation'
        ]

        print(f"      Rendering Manim scene...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=tmpdir,
            timeout=300
        )

        if result.returncode != 0:
            return False, f"Manim render failed:\n{result.stderr[-2000:]}"

        # Find the output file
        rendered = None
        for root, dirs, files in os.walk(os.path.join(tmpdir, 'media')):
            for f in files:
                if f.endswith('.mp4'):
                    rendered = os.path.join(root, f)
                    break
            if rendered:
                break

        if not rendered or not os.path.exists(rendered):
            return False, "Manim produced no output file"

        # Validate output
        info = get_video_info(rendered)
        if not info.get('duration'):
            return False, "Could not read duration of rendered video"

        duration_diff = abs(info['duration'] - total_duration)
        if duration_diff > 2.0:
            print(f"      Warning: Duration mismatch: {info['duration']:.1f}s vs expected {total_duration:.1f}s (diff: {duration_diff:.1f}s)")

        width = info.get('width', 0)
        height = info.get('height', 0)
        if width != 1920 or height != 1080:
            print(f"      Warning: Resolution {width}x{height}, expected 1920x1080")

        # Copy to output
        shutil.copy2(rendered, output_path)
        return True, f"Rendered {info['duration']:.1f}s video ({width}x{height})"


def process_frame(
    video_folder: str,
    frame_num: int,
    math_data: Dict,
    script_data,
    claude: ClaudeClient,
    system_prompt: str,
    force: bool = False,
) -> Tuple[bool, str]:
    """
    Process a single frame: transcribe, compute anchors, generate + render Manim.

    Returns (success, message).
    """
    frames_dir = os.path.join(video_folder, 'frames')
    audio_dir = os.path.join(video_folder, 'audio')
    output_path = os.path.join(frames_dir, f"frame_{frame_num}.mp4")

    # Skip if already exists (unless force)
    if os.path.exists(output_path) and not force:
        return True, f"Frame {frame_num}: Skipped (frame_{frame_num}.mp4 exists)"

    # Get frame data
    frame_info = math_data.get("frames", {}).get(str(frame_num), {})
    narration = frame_info.get("natural_narration", "")
    math_steps = frame_info.get("math_steps", [])
    visual_desc = frame_info.get("original_narration", narration)

    # Get visual description from script
    script_frame = script_data.get_frame(frame_num)
    if script_frame:
        visual_desc = script_frame.visual.reference

    # Get audio duration
    audio_path = os.path.join(audio_dir, f"frame_{frame_num}.mp3")
    if not os.path.exists(audio_path):
        return False, f"Frame {frame_num}: Missing audio file"

    total_duration = get_audio_duration_ffprobe(audio_path)
    print(f"\n    Frame {frame_num}: {len(math_steps)} math steps, {total_duration:.1f}s audio")

    # Step 1: Whisper transcription + alignment to ground-truth narration
    print(f"    [1/3] Getting word-level timestamps...")
    whisper_result = transcribe_with_whisper(audio_path)
    aligned_words = align_words_to_script(narration, whisper_result)
    word_transcript = format_word_transcript(aligned_words)
    print(f"      Aligned {len(aligned_words)} words with timestamps")

    # Step 2: Generate Manim code via Claude (or reuse existing .py)
    code_path = os.path.join(frames_dir, f"frame_{frame_num}_manim.py")
    if os.path.exists(code_path) and not force:
        print(f"    [2/3] Reusing existing frame_{frame_num}_manim.py (use --force to regenerate)")
        with open(code_path, 'r', encoding='utf-8') as f:
            scene_code = f.read()
    else:
        layout = detect_layout(visual_desc, narration, math_steps)
        layout_name = LAYOUT_DESCRIPTIONS[layout].split(' — ')[1] if ' — ' in LAYOUT_DESCRIPTIONS[layout] else LAYOUT_DESCRIPTIONS[layout]
        print(f"    [2/3] Generating Manim code via Claude (Layout {layout}: {layout_name})...")
        user_prompt = build_claude_prompt(
            narration=narration,
            math_steps=math_steps,
            visual_desc=visual_desc,
            total_duration=total_duration,
            frame_number=frame_num,
            word_transcript=word_transcript,
        )
        scene_code = generate_manim_code(claude, system_prompt, user_prompt)
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(scene_code)
        print(f"      Saved Manim code to frame_{frame_num}_manim.py")

    # Step 3: Render
    print(f"    [3/3] Rendering animation...")
    success, message = render_manim_scene(scene_code, output_path, total_duration)

    if success:
        print(f"      {message}")
        return True, f"Frame {frame_num}: {message}"
    else:
        print(f"      FAILED: {message}")
        # Clean up failed output
        if os.path.exists(output_path):
            os.remove(output_path)
        return False, f"Frame {frame_num}: {message}"


def main():
    parser = argparse.ArgumentParser(
        description='Generate Manim math animations for qualifying frames'
    )
    parser.add_argument('video_folder', help='Path to Video-N folder')
    parser.add_argument('--frame', type=int, help='Process specific frame number only')
    parser.add_argument('--force', action='store_true', help='Regenerate even if .mp4 exists')

    args = parser.parse_args()

    # Resolve path
    video_folder = args.video_folder
    if not os.path.isabs(video_folder):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_folder = os.path.join(base_dir, video_folder)

    if not os.path.exists(video_folder):
        print(f"Error: Video folder not found: {video_folder}")
        sys.exit(1)

    print("=" * 70)
    print("MANIM MATH ANIMATION GENERATOR")
    print("=" * 70)
    print(f"Video: {video_folder}")
    print(f"Frame: {args.frame or 'all qualifying'}")
    print(f"Force: {args.force}")

    # Load math verification data
    math_data = load_math_verification(video_folder)
    if not math_data:
        print("\nNo math_verification.json found. Nothing to animate.")
        sys.exit(0)

    # Load script data
    script_data = load_script(Path(video_folder))

    # Select frames for animation
    if args.frame is not None:
        # Manual override — validate the specific frame qualifies
        qualifying = get_qualifying_frames(video_folder, args.frame)
        if not qualifying:
            print(f"\nFrame {args.frame} does not qualify (needs verification + math_steps).")
            sys.exit(0)
        frame_info_map = {}
    else:
        # Smart selection based on audio duration
        qualifying, frame_info_map = select_frames_for_animation(
            video_folder, math_data, script_data
        )

        # Print selection reasoning
        if frame_info_map:
            print(f"\n  Frame selection (threshold: {ANIMATION_DURATION_THRESHOLD}s):")
            for frame_num in sorted(frame_info_map.keys()):
                info = frame_info_map[frame_num]
                tag = "ANIMATE" if frame_num in qualifying else "static"
                title_tag = " [title]" if info['is_title'] else ""
                print(f"    Frame {frame_num}: {info['duration']:.0f}s, "
                      f"{info['math_steps']} steps → {tag}{title_tag}")
            print(f"\n  Animating {len(qualifying)} of {len(frame_info_map)} math frames")

    if not qualifying:
        print("\nNo frames selected for animation.")
        # Write marker even when 0 frames animated (selection decided none needed)
        marker = os.path.join(video_folder, 'frames', '.animate_done')
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        Path(marker).touch()
        sys.exit(0)

    print(f"\nSelected frames: {qualifying}")

    # Initialize Claude client and system prompt
    claude = ClaudeClient()
    system_prompt = load_system_prompt()

    # Process each qualifying frame
    results = []
    for frame_num in qualifying:
        try:
            success, message = process_frame(
                video_folder, frame_num, math_data, script_data,
                claude, system_prompt, args.force
            )
            results.append((frame_num, success, message))
        except Exception as e:
            print(f"\n    Frame {frame_num}: EXCEPTION - {e}")
            import traceback
            traceback.print_exc()
            results.append((frame_num, False, f"Frame {frame_num}: Exception - {e}"))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success_count = 0
    fail_count = 0
    skip_count = 0

    for frame_num, success, message in results:
        if success:
            if "Skipped" in message:
                skip_count += 1
                print(f"  - {message}")
            else:
                success_count += 1
                print(f"  + {message}")
        else:
            fail_count += 1
            print(f"  x {message}")

    print(f"\nAnimated: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed: {fail_count}")

    # Write completion marker (even with skips — the selection was made)
    if fail_count == 0:
        marker = os.path.join(video_folder, 'frames', '.animate_done')
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        Path(marker).touch()

    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
