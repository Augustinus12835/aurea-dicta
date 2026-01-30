#!/usr/bin/env python3
"""
Subtitle Generation Script for Aurea Dicta
Generates SRT subtitles using Whisper for precise timing alignment.

For math videos, uses natural_narration from math_verification.json
to ensure subtitles match TTS audio (e.g., "square root of x" not "sqrt(x)").

Usage:
    python generate_subtitles.py pipeline/LECTURE
    python generate_subtitles.py pipeline/LECTURE --video 3
    python generate_subtitles.py pipeline/LECTURE --force

Output:
    - Video-N/subtitles.srt (separate subtitle file for each video)
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import whisper
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Add parent to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.script_parser import load_script


class FrameData:
    """Data structure for a single frame"""
    def __init__(self, number: int, start_time: float, end_time: float,
                 words: int, narration: str):
        self.number = number
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        self.words = words
        self.narration = narration
        self.audio_path = None
        self.actual_audio_duration = None
        self.actual_start_time = None
        self.actual_end_time = None
        self.whisper_segments = None
        self.aligned_words = None


def parse_script_from_dir(video_folder: str) -> List[FrameData]:
    """
    Parse script file (JSON or MD) to extract frame timing and narration.

    Uses the shared script_parser utility for consistent parsing.
    """
    script_data = load_script(Path(video_folder))

    frames = []
    for frame in script_data.frames:
        frame_data = FrameData(
            number=frame.number,
            start_time=frame.start_seconds,
            end_time=frame.end_seconds,
            words=frame.word_count,
            narration=frame.narration  # Already clean - no visual annotations in JSON
        )
        frames.append(frame_data)

    return frames


def get_audio_duration_ffprobe(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe"""
    import subprocess
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def load_math_verification(video_folder: str) -> Optional[Dict]:
    """Load math_verification.json if it exists."""
    verification_path = os.path.join(video_folder, "math_verification.json")
    if not os.path.exists(verification_path):
        return None
    try:
        with open(verification_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def get_natural_narration(frame_num: int, original_text: str, math_data: Optional[Dict]) -> str:
    """
    Get the best narration text for a frame.

    Prefers natural_narration from math_verification.json if available,
    otherwise falls back to the original text.
    """
    if math_data:
        frame_key = str(frame_num)
        frame_data = math_data.get("frames", {}).get(frame_key)
        if frame_data:
            natural = frame_data.get("natural_narration")
            if natural and frame_data.get("verification_status") in ("correct", "corrected"):
                return natural

    return original_text


def transcribe_audio_with_whisper(audio_path: str, frame_start_time: float,
                                  model_name: str = "small") -> Dict:
    """
    Transcribe audio file using Whisper to get word-level timestamps

    Args:
        audio_path: Path to audio file
        frame_start_time: Start time of this frame in the final video
        model_name: Whisper model to use (tiny, base, small, medium, large)

    Returns:
        Dictionary with segments containing word-level timestamps
    """
    print(f"      Transcribing {os.path.basename(audio_path)} with Whisper...")

    # Load model (cached after first use)
    model = whisper.load_model(model_name)

    # Transcribe with word-level timestamps
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en"
    )

    # Adjust timestamps to be relative to video start
    for segment in result.get('segments', []):
        segment['start'] += frame_start_time
        segment['end'] += frame_start_time

        if 'words' in segment:
            for word in segment['words']:
                word['start'] += frame_start_time
                word['end'] += frame_start_time

    return result


def align_script_to_whisper_timestamps(script_text: str, whisper_result: Dict) -> List[Dict]:
    """
    Align actual script text with Whisper word timestamps

    This corrects Whisper transcription errors while preserving precise timing.
    """
    # Extract all Whisper words with timestamps
    whisper_words = []
    for segment in whisper_result.get('segments', []):
        if 'words' in segment:
            whisper_words.extend(segment['words'])

    # Clean and tokenize script text
    script_words = script_text.replace('\n', ' ').split()

    aligned_words = []

    # If counts match, direct mapping
    if len(script_words) == len(whisper_words):
        for script_word, whisper_word in zip(script_words, whisper_words):
            aligned_words.append({
                'word': script_word,
                'start': whisper_word['start'],
                'end': whisper_word['end']
            })
    else:
        # Counts don't match - use proportional mapping
        if not whisper_words:
            return []

        total_duration = whisper_words[-1]['end'] - whisper_words[0]['start']
        time_per_word = total_duration / len(script_words)

        current_time = whisper_words[0]['start']
        for script_word in script_words:
            aligned_words.append({
                'word': script_word,
                'start': current_time,
                'end': current_time + time_per_word
            })
            current_time += time_per_word

    return aligned_words


def convert_to_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_subtitles_from_corrected_timestamps(frames: List[FrameData], output_path: str,
                                                max_chars_per_line: int = 42) -> int:
    """
    Generate SRT subtitle file using corrected script text with Whisper timestamps

    Returns number of subtitle entries created
    """
    subtitle_entries = []
    entry_id = 1

    for frame in frames:
        if not hasattr(frame, 'aligned_words') or not frame.aligned_words:
            continue

        # Group words into subtitle chunks (max 2 lines, max chars per line)
        words = frame.aligned_words
        current_chunk = []
        current_line = []
        current_length = 0
        chunk_start_time = None

        for word_info in words:
            word = word_info['word'].strip()
            if not word:
                continue

            if chunk_start_time is None:
                chunk_start_time = word_info['start']

            word_length = len(word)

            # Check if adding this word exceeds line length
            if current_length + word_length + (1 if current_line else 0) > max_chars_per_line:
                # Start new line
                if current_line:
                    current_chunk.append(' '.join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    # Word too long, add anyway
                    current_chunk.append(word)
                    current_line = []
                    current_length = 0

                # Check if we've filled 2 lines (create subtitle entry)
                if len(current_chunk) >= 2:
                    text = '\n'.join(current_chunk)
                    start_ts = convert_to_srt_timestamp(chunk_start_time)
                    end_ts = convert_to_srt_timestamp(word_info['end'])
                    subtitle_entries.append(f"{entry_id}\n{start_ts} --> {end_ts}\n{text}\n")
                    entry_id += 1

                    current_chunk = []
                    chunk_start_time = None
                    if current_line:
                        chunk_start_time = word_info['start']
            else:
                current_line.append(word)
                current_length += word_length + (1 if len(current_line) > 1 else 0)

        # Add remaining words as final subtitle entry
        if current_line:
            current_chunk.append(' '.join(current_line))

        if current_chunk and chunk_start_time is not None:
            text = '\n'.join(current_chunk)
            start_ts = convert_to_srt_timestamp(chunk_start_time)
            end_ts = convert_to_srt_timestamp(words[-1]['end'])
            subtitle_entries.append(f"{entry_id}\n{start_ts} --> {end_ts}\n{text}\n")
            entry_id += 1

    # Write SRT file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(subtitle_entries))

    return len(subtitle_entries)


def calculate_actual_frame_times(frames: List[FrameData]) -> None:
    """Calculate actual frame start/end times based on measured audio durations"""
    current_time = 0.0

    for frame in frames:
        frame.actual_start_time = current_time
        frame.actual_end_time = current_time + frame.actual_audio_duration
        current_time = frame.actual_end_time


def generate_subtitles_for_video(video_folder: str, force: bool = False) -> Tuple[bool, str]:
    """
    Generate subtitles for a single video folder.

    Returns (success, message)
    """
    video_name = os.path.basename(video_folder)

    # Check if subtitles already exist
    subtitle_path = os.path.join(video_folder, 'subtitles.srt')
    if os.path.exists(subtitle_path) and not force:
        return True, f"{video_name}: Skipped (subtitles.srt exists)"

    # Check required files
    json_path = os.path.join(video_folder, 'script.json')
    md_path = os.path.join(video_folder, 'script.md')
    audio_dir = os.path.join(video_folder, 'audio')

    if not os.path.exists(json_path) and not os.path.exists(md_path):
        return False, f"{video_name}: Missing script.json or script.md"

    if not os.path.exists(audio_dir):
        return False, f"{video_name}: Missing audio/ directory"

    print(f"\n{'='*60}")
    print(f"Generating subtitles: {video_name}")
    print('='*60)

    try:
        # Parse script
        print("\n[1/4] Parsing script...")
        frames = parse_script_from_dir(video_folder)
        print(f"      Found {len(frames)} frames")

        # Load math verification for natural narration
        math_data = load_math_verification(video_folder)
        if math_data:
            verified_count = len(math_data.get("frames", {}))
            print(f"      Math verification available ({verified_count} frames)")

        # Validate audio files and get durations
        print("\n[2/4] Validating audio files...")
        for frame in frames:
            audio_name = f"frame_{frame.number}.mp3"
            audio_path = os.path.join(audio_dir, audio_name)

            if not os.path.exists(audio_path):
                return False, f"{video_name}: Missing {audio_name}"

            frame.audio_path = audio_path
            frame.actual_audio_duration = get_audio_duration_ffprobe(audio_path)

            # Apply natural narration for math videos
            frame.narration = get_natural_narration(frame.number, frame.narration, math_data)

        calculate_actual_frame_times(frames)
        total_duration = sum(f.actual_audio_duration for f in frames)
        print(f"      Total audio duration: {total_duration:.1f}s")

        # Transcribe with Whisper
        print("\n[3/4] Transcribing with Whisper...")
        for frame in frames:
            frame.whisper_segments = transcribe_audio_with_whisper(
                frame.audio_path,
                frame.actual_start_time,
                model_name="small"
            )
            # Align script text to Whisper timestamps
            frame.aligned_words = align_script_to_whisper_timestamps(
                frame.narration,
                frame.whisper_segments
            )

        print(f"      Transcribed all {len(frames)} audio files")

        # Generate SRT file
        print("\n[4/4] Generating subtitles.srt...")
        num_subtitles = generate_subtitles_from_corrected_timestamps(frames, subtitle_path)
        print(f"      Created {num_subtitles} subtitle entries")

        return True, f"{video_name}: Generated {num_subtitles} subtitles"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"{video_name}: Error - {e}"


def find_video_folders(lecture_folder: str, specific_video: Optional[int] = None) -> List[str]:
    """Find all Video-N folders in a lecture folder"""
    video_folders = []

    for item in sorted(os.listdir(lecture_folder)):
        if item.startswith('Video-'):
            video_num = int(item.split('-')[1])

            if specific_video is not None and video_num != specific_video:
                continue

            video_path = os.path.join(lecture_folder, item)
            if os.path.isdir(video_path):
                video_folders.append(video_path)

    return video_folders


def main():
    parser = argparse.ArgumentParser(
        description='Generate subtitles for Aurea Dicta videos using Whisper'
    )
    parser.add_argument('lecture_folder', help='Path to lecture folder (e.g., pipeline/LECTURE)')
    parser.add_argument('--video', type=int, help='Generate for specific video number only')
    parser.add_argument('--force', action='store_true', help='Regenerate even if subtitles.srt exists')

    args = parser.parse_args()

    # Resolve path
    lecture_folder = args.lecture_folder
    if not os.path.isabs(lecture_folder):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lecture_folder = os.path.join(base_dir, lecture_folder)

    if not os.path.exists(lecture_folder):
        print(f"Error: Lecture folder not found: {lecture_folder}")
        sys.exit(1)

    # Find video folders
    video_folders = find_video_folders(lecture_folder, args.video)

    if not video_folders:
        print(f"Error: No Video-N folders found in {lecture_folder}")
        sys.exit(1)

    print("="*60)
    print("AUREA DICTA SUBTITLE GENERATION")
    print("="*60)
    print(f"Lecture: {os.path.basename(lecture_folder)}")
    print(f"Videos: {len(video_folders)}")
    print(f"Force: {args.force}")
    print("="*60)

    # Process each video
    results = []
    for video_folder in video_folders:
        success, message = generate_subtitles_for_video(video_folder, args.force)
        results.append((success, message))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for success, message in results:
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

    print()
    print(f"Generated: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed: {fail_count}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
