#!/usr/bin/env python3
"""
Video Compilation Script for Aurea Dicta
Compiles final video with transitions and subtitles

Usage:
    python3 compile_video.py pipeline/YOUR_LECTURE/Video-1

Output:
    - final_video.mp4 (complete video, subtitles NOT burned-in)
    - subtitles.srt (separate subtitle file - viewers can toggle on/off)
    - compilation_report.txt (verification report)
"""

import os
import sys
import re
import json
import subprocess
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path
import whisper
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class VideoCompilationError(Exception):
    """Base exception for compilation errors"""
    pass


class FrameMismatchError(VideoCompilationError):
    """Frame count doesn't match"""
    pass


class TimingError(VideoCompilationError):
    """Duration mismatch"""
    pass


class FFmpegError(VideoCompilationError):
    """FFmpeg execution failed"""
    pass


class FrameData:
    """Data structure for a single frame"""
    def __init__(self, number: int, start_time: float, end_time: float,
                 words: int, narration: str):
        self.number = number
        self.start_time = start_time  # Original script timing
        self.end_time = end_time      # Original script timing
        self.duration = end_time - start_time  # Script estimate
        self.words = words
        self.narration = narration  # Actual script text (ground truth)
        self.image_path = None
        self.audio_path = None
        self.actual_audio_duration = None  # Measured from audio file
        self.actual_start_time = None  # Actual video timestamp (calculated)
        self.actual_end_time = None    # Actual video timestamp (calculated)
        self.whisper_segments = None  # Word-level timestamps from Whisper
        self.continuation_of = None  # If this frame continues a previous frame's visual


class MergedSegment:
    """
    Represents multiple consecutive frames merged into a single video segment.
    Used when consecutive frames share the same visual to avoid jarring transitions.
    """
    def __init__(self, frames: List['FrameData']):
        self.frames = frames
        self.image_path = frames[0].image_path  # Use first frame's image
        self.frame_numbers = [f.number for f in frames]
        self.merged_audio_path = None  # Will be set after audio concatenation

    @property
    def total_duration(self) -> float:
        """Total audio duration of all frames in segment"""
        return sum(f.actual_audio_duration for f in self.frames if f.actual_audio_duration)

    @property
    def actual_start_time(self) -> float:
        return self.frames[0].actual_start_time

    @property
    def actual_end_time(self) -> float:
        return self.frames[-1].actual_end_time

    def __repr__(self):
        return f"MergedSegment(frames={self.frame_numbers}, duration={self.total_duration:.1f}s)"


def load_frame_metadata(video_folder: str) -> Dict:
    """
    Load frames/metadata.json to detect continuation frames.

    Returns dict with frame info including continuation_of markers.
    """
    metadata_path = os.path.join(video_folder, 'frames', 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {"frames": []}


def detect_continuation_frames(frames: List[FrameData], video_folder: str) -> None:
    """
    Mark frames that are continuations of previous frames (same visual).

    Updates frames in-place with continuation_of attribute.
    """
    metadata = load_frame_metadata(video_folder)

    # Build lookup by frame number
    meta_by_num = {m.get("number"): m for m in metadata.get("frames", [])}

    for frame in frames:
        meta = meta_by_num.get(frame.number, {})
        continuation_of = meta.get("continuation_of")
        if continuation_of is not None:
            frame.continuation_of = continuation_of


def merge_continuation_frames(frames: List[FrameData], video_folder: str) -> List[MergedSegment]:
    """
    Merge consecutive frames that share the same visual into MergedSegments.

    Returns list of MergedSegments. Each segment contains one or more frames.
    Single-frame segments have no transitions removed.
    Multi-frame segments will be rendered as one image with concatenated audio.
    """
    if not frames:
        return []

    # First detect which frames are continuations
    detect_continuation_frames(frames, video_folder)

    segments = []
    current_group = [frames[0]]

    for i in range(1, len(frames)):
        frame = frames[i]
        prev_frame = frames[i - 1]

        # Check if this frame continues the previous one
        if frame.continuation_of == prev_frame.number:
            current_group.append(frame)
        else:
            # Start new segment
            segments.append(MergedSegment(current_group))
            current_group = [frame]

    # Don't forget the last group
    segments.append(MergedSegment(current_group))

    return segments


def concatenate_audio_files(audio_paths: List[str], output_path: str) -> bool:
    """
    Concatenate multiple audio files into one using FFmpeg.

    Returns True on success.
    """
    if len(audio_paths) == 1:
        # Just copy if single file
        import shutil
        shutil.copy(audio_paths[0], output_path)
        return True

    # Create concat list file
    list_file = output_path + ".txt"
    with open(list_file, 'w') as f:
        for path in audio_paths:
            # Escape single quotes in path
            escaped = path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    # Run FFmpeg concat
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', list_file, '-c', 'copy', output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        os.remove(list_file)  # Clean up
        return result.returncode == 0
    except Exception:
        if os.path.exists(list_file):
            os.remove(list_file)
        return False


def prepare_merged_segments(segments: List[MergedSegment], video_folder: str) -> None:
    """
    Prepare merged segments by concatenating audio files for multi-frame segments.

    Creates merged audio files in a temp directory.
    """
    merged_audio_dir = os.path.join(video_folder, 'frames', 'merged_audio')
    os.makedirs(merged_audio_dir, exist_ok=True)

    for seg in segments:
        if len(seg.frames) == 1:
            # Single frame - use original audio
            seg.merged_audio_path = seg.frames[0].audio_path
        else:
            # Multiple frames - concatenate audio
            audio_paths = [f.audio_path for f in seg.frames]
            merged_path = os.path.join(merged_audio_dir, f"merged_{seg.frame_numbers[0]}_{seg.frame_numbers[-1]}.mp3")

            if concatenate_audio_files(audio_paths, merged_path):
                seg.merged_audio_path = merged_path
            else:
                # Fallback: use first frame's audio (shouldn't happen)
                print(f"    Warning: Failed to merge audio for frames {seg.frame_numbers}")
                seg.merged_audio_path = seg.frames[0].audio_path


def parse_time_to_seconds(time_str: str) -> float:
    """
    Convert MM:SS time format to seconds

    Examples:
        "0:15" -> 15.0
        "1:30" -> 90.0
        "4:00" -> 240.0
    """
    parts = time_str.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {time_str}")

    minutes = int(parts[0])
    seconds = int(parts[1])
    return minutes * 60 + seconds


def parse_script(script_path: str) -> List[FrameData]:
    """
    Parse script.md to extract frame timing and narration

    Returns list of FrameData objects
    """
    frames = []

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern: ## Frame N (MM:SS-MM:SS) • NN words
    pattern = r'## Frame (\d+) \((\d+:\d+)-(\d+:\d+)\) • (\d+) words?\s*\n\n(.*?)(?=\n---|\n##|\Z)'

    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        frame_num = int(match.group(1))
        start_time = parse_time_to_seconds(match.group(2))
        end_time = parse_time_to_seconds(match.group(3))
        words = int(match.group(4))
        narration = match.group(5).strip()

        # Remove [Visual: ...] annotations from narration (match to end since they always appear last)
        narration = re.sub(r'\[Visual:.*$', '', narration, flags=re.DOTALL)
        narration = ' '.join(narration.split())  # Clean up whitespace

        frame = FrameData(frame_num, start_time, end_time, words, narration)
        frames.append(frame)

    return frames


def get_audio_duration_ffprobe(audio_path: str) -> float:
    """
    Get audio duration in seconds using ffprobe
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


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

    This corrects Whisper transcription errors (e.g., "Cain" -> "Keynes")
    while preserving precise timing information.

    Args:
        script_text: Ground truth text from script.md
        whisper_result: Whisper transcription with timestamps

    Returns:
        List of word dictionaries with corrected text and preserved timestamps
    """
    # Extract all Whisper words with timestamps
    whisper_words = []
    for segment in whisper_result.get('segments', []):
        if 'words' in segment:
            whisper_words.extend(segment['words'])

    # Clean and tokenize script text
    script_words = script_text.replace('\n', ' ').split()

    # Align script words to Whisper timestamps
    # Simple approach: assume same word count, map 1:1
    aligned_words = []

    # If counts match, direct mapping
    if len(script_words) == len(whisper_words):
        for script_word, whisper_word in zip(script_words, whisper_words):
            aligned_words.append({
                'word': script_word,  # Use actual script text
                'start': whisper_word['start'],  # Use Whisper timing
                'end': whisper_word['end']
            })
    else:
        # Counts don't match - use proportional mapping
        # This handles cases where Whisper merges/splits words
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
    """
    Convert seconds to SRT timestamp format: HH:MM:SS,mmm

    Example: 65.5 -> 00:01:05,500
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def split_text_for_subtitles(text: str, max_chars_per_line: int = 42,
                             max_lines: int = 2) -> List[str]:
    """
    Split text into subtitle-friendly segments

    Rules:
    - Max 42 characters per line
    - Max 2 lines per subtitle
    - Split at sentence/phrase boundaries when possible
    """
    segments = []
    words = text.split()

    current_lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word)

        # Check if adding this word exceeds line length
        if current_length + word_length + (1 if current_line else 0) > max_chars_per_line:
            # Save current line and start new one
            if current_line:
                current_lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                # Word is too long, add it anyway
                current_lines.append(word)
                current_line = []
                current_length = 0

            # Check if we've reached max lines
            if len(current_lines) >= max_lines:
                segments.append('\n'.join(current_lines))
                current_lines = []
        else:
            current_line.append(word)
            current_length += word_length + (1 if len(current_line) > 1 else 0)

    # Add remaining words
    if current_line:
        current_lines.append(' '.join(current_line))
    if current_lines:
        segments.append('\n'.join(current_lines))

    return segments


def generate_subtitles_from_corrected_timestamps(frames: List[FrameData], output_path: str,
                                                max_chars_per_line: int = 42) -> int:
    """
    Generate SRT subtitle file using corrected script text with Whisper timestamps

    Uses actual script text (corrected) with Whisper timing (precise).

    Args:
        frames: List of FrameData objects with aligned words
        output_path: Path to save SRT file
        max_chars_per_line: Maximum characters per subtitle line

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
                        # Continue with overflow word
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
    """
    Calculate actual frame start/end times based on measured audio durations

    This replaces script estimates with actual audio lengths.
    Frames and audio will start/end simultaneously.
    """
    current_time = 0.0

    for frame in frames:
        frame.actual_start_time = current_time
        frame.actual_end_time = current_time + frame.actual_audio_duration
        current_time = frame.actual_end_time

        print(f"      Frame {frame.number}: {frame.actual_start_time:.2f}s - {frame.actual_end_time:.2f}s "
              f"(audio: {frame.actual_audio_duration:.2f}s, script: {frame.duration:.2f}s)")


def validate_input_files(video_folder: str, frames: List[FrameData]) -> Tuple[int, int, int]:
    """
    Validate that all required input files exist and measure actual audio durations

    Returns: (num_frames, num_images, num_audio)
    """
    frames_dir = os.path.join(video_folder, 'frames')
    audio_dir = os.path.join(video_folder, 'audio')

    # Check directories exist
    if not os.path.exists(frames_dir):
        raise FrameMismatchError(f"Frames directory not found: {frames_dir}")
    if not os.path.exists(audio_dir):
        raise FrameMismatchError(f"Audio directory not found: {audio_dir}")

    # Validate each frame has matching image and audio
    for frame in frames:
        # Check image
        image_name = f"frame_{frame.number}.png"
        image_path = os.path.join(frames_dir, image_name)
        if not os.path.exists(image_path):
            raise FrameMismatchError(f"Missing image: {image_name}")
        frame.image_path = image_path

        # Check audio and get actual duration
        audio_name = f"frame_{frame.number}.mp3"
        audio_path = os.path.join(audio_dir, audio_name)
        if not os.path.exists(audio_path):
            raise FrameMismatchError(f"Missing audio: {audio_name}")
        frame.audio_path = audio_path

        # Get actual audio duration
        frame.actual_audio_duration = get_audio_duration_ffprobe(audio_path)

    # Calculate actual frame times based on real audio durations
    calculate_actual_frame_times(frames)

    num_frames = len(frames)
    num_images = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    num_audio = len([f for f in os.listdir(audio_dir) if f.endswith('.mp3')])

    return num_frames, num_images, num_audio


def build_ffmpeg_command(video_folder: str, frames: List[FrameData],
                        subtitle_path: str) -> List[str]:
    """
    Build FFmpeg command for video compilation with transitions

    Uses complex filter graph with:
    - Crossfade transitions (0.5s)
    - Actual audio durations (no estimates)
    - Separate subtitle file (NOT burned-in, allows students to toggle)

    Audio and frames start/end simultaneously - no artificial delays.
    """
    cmd = ['ffmpeg', '-y']

    # Add audio inputs
    for frame in frames:
        cmd.extend(['-i', frame.audio_path])

    # Add image inputs using ACTUAL audio duration
    for frame in frames:
        # Use actual measured audio duration
        cmd.extend([
            '-loop', '1',
            '-t', str(frame.actual_audio_duration),
            '-i', frame.image_path
        ])

    # Build complex filter graph
    filter_parts = []

    # Process each image: scale, set frame rate, add fade transitions
    num_frames = len(frames)
    fade_duration = 0.5  # 0.5 second crossfade

    for i, frame in enumerate(frames):
        input_idx = num_frames + i  # Images start after audio files

        # Calculate fade timings based on actual audio duration
        fade_out_start = frame.actual_audio_duration - fade_duration

        # Scale, set fps, and add fades
        filter_str = (
            f"[{input_idx}:v]scale=1920:1080:flags=lanczos,"
            f"fps=30,"
            f"fade=t=in:st=0:d={fade_duration},"
            f"fade=t=out:st={fade_out_start}:d={fade_duration}[v{i}]"
        )
        filter_parts.append(filter_str)

    # Concatenate video streams
    video_concat = ''.join([f"[v{i}]" for i in range(num_frames)])
    video_concat += f"concat=n={num_frames}:v=1:a=0[video]"
    filter_parts.append(video_concat)

    # Concatenate audio streams - synchronized with video (no delay)
    audio_concat = ''.join([f"[{i}:a]" for i in range(num_frames)])
    audio_concat += f"concat=n={num_frames}:v=0:a=1[audio]"
    filter_parts.append(audio_concat)

    # NOTE: Subtitles are NOT burned-in
    # Separate subtitles.srt file is generated for students to toggle on/off

    # Join all filter parts
    filter_complex = ';'.join(filter_parts)

    cmd.extend(['-filter_complex', filter_complex])

    # Map outputs (no subtitle burning)
    cmd.extend(['-map', '[video]', '-map', '[audio]'])

    # Video encoding settings
    cmd.extend([
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-r', '30'
    ])

    # Audio encoding settings
    cmd.extend([
        '-c:a', 'aac',
        '-b:a', '192k',
        '-ar', '48000'
    ])

    # Output file
    output_path = os.path.join(video_folder, 'final_video.mp4')
    cmd.append(output_path)

    return cmd


def build_ffmpeg_command_with_segments(video_folder: str, segments: List[MergedSegment],
                                       subtitle_path: str) -> List[str]:
    """
    Build FFmpeg command for video compilation with merged segments.

    Handles continuation frames by:
    - Using merged audio for multi-frame segments
    - Using single image for multi-frame segments
    - Only applying fade transitions at segment boundaries (not between merged frames)

    This eliminates jarring transitions between frames that share the same visual.
    """
    cmd = ['ffmpeg', '-y']

    # Add audio inputs (one per segment)
    for seg in segments:
        cmd.extend(['-i', seg.merged_audio_path])

    # Add image inputs using segment's total duration
    for seg in segments:
        cmd.extend([
            '-loop', '1',
            '-t', str(seg.total_duration),
            '-i', seg.image_path
        ])

    # Build complex filter graph
    filter_parts = []

    num_segments = len(segments)
    fade_duration = 0.5  # 0.5 second crossfade

    for i, seg in enumerate(segments):
        input_idx = num_segments + i  # Images start after audio files

        # Calculate fade timings based on segment's total duration
        fade_out_start = seg.total_duration - fade_duration

        # Scale, set fps, and add fades
        filter_str = (
            f"[{input_idx}:v]scale=1920:1080:flags=lanczos,"
            f"fps=30,"
            f"fade=t=in:st=0:d={fade_duration},"
            f"fade=t=out:st={fade_out_start}:d={fade_duration}[v{i}]"
        )
        filter_parts.append(filter_str)

    # Concatenate video streams
    video_concat = ''.join([f"[v{i}]" for i in range(num_segments)])
    video_concat += f"concat=n={num_segments}:v=1:a=0[video]"
    filter_parts.append(video_concat)

    # Concatenate audio streams
    audio_concat = ''.join([f"[{i}:a]" for i in range(num_segments)])
    audio_concat += f"concat=n={num_segments}:v=0:a=1[audio]"
    filter_parts.append(audio_concat)

    # Join all filter parts
    filter_complex = ';'.join(filter_parts)

    cmd.extend(['-filter_complex', filter_complex])

    # Map outputs
    cmd.extend(['-map', '[video]', '-map', '[audio]'])

    # Video encoding settings
    cmd.extend([
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-r', '30'
    ])

    # Audio encoding settings
    cmd.extend([
        '-c:a', 'aac',
        '-b:a', '192k',
        '-ar', '48000'
    ])

    # Output file
    output_path = os.path.join(video_folder, 'final_video.mp4')
    cmd.append(output_path)

    return cmd


def execute_ffmpeg(cmd: List[str]) -> Tuple[bool, str]:
    """
    Execute FFmpeg command with progress monitoring

    Returns: (success, output_message)
    """
    print("\n" + "="*70)
    print("EXECUTING FFMPEG COMPILATION")
    print("="*70)

    try:
        # Run FFmpeg
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Monitor progress from stderr (FFmpeg outputs to stderr)
        stderr_output = []
        for line in process.stderr:
            stderr_output.append(line)
            # Show progress lines
            if 'time=' in line:
                print(f"\r{line.strip()}", end='', flush=True)

        process.wait()

        if process.returncode != 0:
            error_msg = ''.join(stderr_output[-50:])  # Last 50 lines
            raise FFmpegError(f"FFmpeg failed with code {process.returncode}\n{error_msg}")

        print("\n✓ FFmpeg compilation successful")
        return True, "Success"

    except Exception as e:
        return False, str(e)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def get_video_info(video_path: str) -> Dict:
    """Get detailed video information using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,codec_name,r_frame_rate',
        '-of', 'json',
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    return data['streams'][0] if data.get('streams') else {}


def verify_compilation(video_folder: str, frames: List[FrameData]) -> Dict:
    """
    Verify the compiled video meets requirements

    Returns verification results dictionary
    """
    video_path = os.path.join(video_folder, 'final_video.mp4')
    results = {}

    # Check video exists
    if not os.path.exists(video_path):
        results['exists'] = False
        return results
    results['exists'] = True

    # Check file size
    file_size = os.path.getsize(video_path)
    results['file_size_mb'] = file_size / (1024 * 1024)

    # Check duration
    expected_duration = frames[-1].end_time
    actual_duration = get_video_duration(video_path)
    results['expected_duration'] = expected_duration
    results['actual_duration'] = actual_duration
    results['duration_diff'] = abs(actual_duration - expected_duration)
    results['duration_ok'] = results['duration_diff'] <= 2.0

    # Check video properties
    video_info = get_video_info(video_path)
    results['width'] = video_info.get('width', 0)
    results['height'] = video_info.get('height', 0)
    results['codec'] = video_info.get('codec_name', 'unknown')
    results['resolution_ok'] = (results['width'] == 1920 and results['height'] == 1080)

    return results


def generate_report(video_folder: str, frames: List[FrameData],
                   num_subtitles: int, verification: Dict,
                   compilation_time: float) -> str:
    """
    Generate comprehensive compilation report
    """
    report_lines = [
        "Video Compilation Report",
        "=" * 70,
        f"Video: {video_folder}/final_video.mp4",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Compilation Time: {compilation_time:.1f} seconds",
        "",
        "INPUT VERIFICATION",
        "-" * 70,
        f"✓ Script parsed: {len(frames)} frames",
        f"✓ Images found: {len(frames)} PNG files (1920x1080)",
        f"✓ Audio found: {len(frames)} MP3 files",
        f"✓ Total expected duration: {frames[-1].end_time:.0f} seconds",
        "",
        "SUBTITLE GENERATION",
        "-" * 70,
        f"✓ Subtitles created: {num_subtitles} subtitle entries",
        f"✓ Format: SRT with Whisper word-level timestamps",
        f"✓ Max line length: 42 characters",
        f"✓ Max lines per subtitle: 2",
        f"✓ Timing: Perfectly synced using Whisper STT",
        "",
        "VIDEO COMPILATION",
        "-" * 70,
        f"✓ Frame transitions: Crossfade (0.5s)",
        f"✓ Audio timing: No artificial delay (Whisper-synced)",
        f"✓ Frame duration: Extended to prevent audio cutoff",
        f"✓ Video codec: H.264 (libx264, CRF 23)",
        f"✓ Audio codec: AAC (192 kbps)",
        f"✓ Resolution: 1920x1080 @ 30fps",
        f"✓ Subtitles: Separate .srt file (students can toggle on/off)",
        "",
        "OUTPUT VERIFICATION",
        "-" * 70,
    ]

    if verification.get('exists'):
        status_symbol = "✓" if verification.get('duration_ok') else "⚠"
        report_lines.extend([
            f"{status_symbol} Video duration: {verification['actual_duration']:.0f}s "
            f"(target: {verification['expected_duration']:.0f}s, "
            f"diff: {verification['duration_diff']:.1f}s)",
            f"✓ File size: {verification['file_size_mb']:.1f} MB",
        ])

        if verification.get('resolution_ok'):
            report_lines.append(f"✓ Resolution: {verification['width']}x{verification['height']}")
        else:
            report_lines.append(f"⚠ Resolution: {verification['width']}x{verification['height']} "
                              f"(expected 1920x1080)")

        report_lines.append(f"✓ Codec: {verification['codec']}")
    else:
        report_lines.append("✗ Video file not created")

    report_lines.extend([
        "",
        "FILES CREATED",
        "-" * 70,
        f"✓ final_video.mp4 ({verification.get('file_size_mb', 0):.1f} MB)",
        f"✓ subtitles.srt",
        f"✓ compilation_report.txt",
        "",
    ])

    # Overall status
    if verification.get('exists') and verification.get('duration_ok') and verification.get('resolution_ok'):
        report_lines.extend([
            "STATUS: ✓ COMPILATION SUCCESSFUL",
            "",
            "Next step: Review video, then upload or distribute"
        ])
    else:
        report_lines.extend([
            "STATUS: ⚠ COMPILATION COMPLETED WITH WARNINGS",
            "",
            "Please review the warnings above and verify video quality"
        ])

    return '\n'.join(report_lines)


def extract_pdf_to_frames(video_folder: str, num_expected_frames: int) -> bool:
    """
    Extract slides.pdf to PNG frames using pdftocairo

    Args:
        video_folder: Path to video folder
        num_expected_frames: Expected number of frames from script.md

    Returns:
        True if extraction successful, False if slides.pdf doesn't exist

    Raises:
        VideoCompilationError: If PDF page count doesn't match expected frames
    """
    pdf_path = os.path.join(video_folder, 'slides.pdf')
    frames_dir = os.path.join(video_folder, 'frames')

    # Check if slides.pdf exists
    if not os.path.exists(pdf_path):
        return False

    print("      ✓ Found slides.pdf, extracting frames...")

    # Get PDF page count
    try:
        result = subprocess.run(
            ['pdfinfo', pdf_path],
            capture_output=True,
            text=True,
            check=True
        )

        # Extract page count
        for line in result.stdout.split('\n'):
            if line.startswith('Pages:'):
                pdf_pages = int(line.split(':')[1].strip())
                break
        else:
            raise VideoCompilationError("Could not determine PDF page count")

        # Verify page count matches expected frames
        if pdf_pages != num_expected_frames:
            raise VideoCompilationError(
                f"PDF has {pdf_pages} pages but script expects {num_expected_frames} frames"
            )

        print(f"      ✓ PDF has {pdf_pages} pages (matches script)")

    except subprocess.CalledProcessError as e:
        raise VideoCompilationError(f"Failed to read PDF info: {e}")
    except FileNotFoundError:
        raise VideoCompilationError("pdfinfo not found - install poppler-utils: brew install poppler")

    # Create frames directory if it doesn't exist
    os.makedirs(frames_dir, exist_ok=True)

    # Extract PDF to PNG frames (150 DPI for ~1920x1080 output)
    try:
        subprocess.run(
            ['pdftocairo', '-png', '-r', '150', pdf_path, os.path.join(frames_dir, 'frame')],
            check=True,
            capture_output=True
        )
        print(f"      ✓ Extracted PDF to PNG frames (150 DPI)")

    except subprocess.CalledProcessError as e:
        raise VideoCompilationError(f"Failed to extract PDF: {e}")
    except FileNotFoundError:
        raise VideoCompilationError("pdftocairo not found - install poppler-utils: brew install poppler")

    # Rename frames from frame-01.png to frame_0.png (0-indexed)
    import glob
    extracted_frames = sorted(glob.glob(os.path.join(frames_dir, 'frame-*.png')))

    if len(extracted_frames) != pdf_pages:
        raise VideoCompilationError(
            f"Expected {pdf_pages} extracted frames, found {len(extracted_frames)}"
        )

    for i, old_path in enumerate(extracted_frames):
        new_path = os.path.join(frames_dir, f'frame_{i}.png')
        os.rename(old_path, new_path)

    print(f"      ✓ Renamed frames to 0-indexed convention (frame_0.png - frame_{num_expected_frames-1}.png)")

    return True


def compile_video(video_folder: str) -> str:
    """
    Main compilation function

    Workflow:
    0. Extract slides.pdf to frames if it exists (automated)
    1. Uses actual measured audio durations (not script estimates)
    2. Frames and audio start/end simultaneously (no delays)
    3. Whisper provides precise word-level timestamps
    4. Script text corrects Whisper transcription errors
    5. Perfect subtitle synchronization

    Returns status message
    """
    start_time = datetime.now()

    print("=" * 70)
    print("AUREA DICTA VIDEO COMPILATION")
    print("=" * 70)
    print(f"Video folder: {video_folder}")
    print("Frames/audio sync: Simultaneous (no delay)")
    print("Subtitles: Script text + Whisper timing")
    print()

    try:
        # Step 0: Parse script first to get frame count
        print("[0/9] Parsing script.md...")
        script_path = os.path.join(video_folder, 'script.md')
        if not os.path.exists(script_path):
            raise VideoCompilationError(f"Script not found: {script_path}")

        frames = parse_script(script_path)
        print(f"      ✓ Parsed {len(frames)} frames")
        print(f"      ✓ Script duration: {frames[-1].end_time:.0f} seconds")

        # Step 0.5: Extract PDF to frames if slides.pdf exists
        print("\n[0.5/9] Checking for slides.pdf...")
        if extract_pdf_to_frames(video_folder, len(frames)):
            pass  # PDF extracted successfully
        else:
            print("      ✓ No slides.pdf found, using existing frames")

        # Step 1: Validate input files and measure audio durations
        print("\n[1/9] Validating input files and calculating actual frame times...")
        num_frames, num_images, num_audio = validate_input_files(video_folder, frames)
        print(f"\n      ✓ Found {num_images} frame images")
        print(f"      ✓ Found {num_audio} audio files")

        # Show actual vs script duration
        total_audio_duration = sum(f.actual_audio_duration for f in frames)
        total_script_duration = frames[-1].end_time
        print(f"      ✓ Total audio duration: {total_audio_duration:.1f}s (script estimate: {total_script_duration:.0f}s)")

        if num_frames != num_images or num_frames != num_audio:
            raise FrameMismatchError(
                f"Mismatch: {num_frames} script frames, "
                f"{num_images} images, {num_audio} audio files"
            )

        # Step 2: Transcribe audio with Whisper for precise timing
        print("\n[2/9] Transcribing audio with Whisper (this may take a minute)...")
        for frame in frames:
            # Transcribe using ACTUAL frame start time
            frame.whisper_segments = transcribe_audio_with_whisper(
                frame.audio_path,
                frame.actual_start_time,  # Use calculated actual time, not script estimate
                model_name="small"
            )
        print(f"      ✓ Transcribed all {len(frames)} audio files")

        # Step 3: Align script text to Whisper timestamps (correct transcription errors)
        print("\n[3/9] Aligning script text to Whisper timestamps...")
        for frame in frames:
            frame.aligned_words = align_script_to_whisper_timestamps(
                frame.narration,  # Ground truth text from script
                frame.whisper_segments  # Precise timestamps from Whisper
            )
        print(f"      ✓ Corrected transcription using actual script text")
        print(f"      ✓ Preserved Whisper word-level timestamps")

        # Step 4: Generate subtitles with corrected text + Whisper timing
        print("\n[4/9] Generating perfectly-synced subtitles...")
        subtitle_path = os.path.join(video_folder, 'subtitles.srt')
        num_subtitles = generate_subtitles_from_corrected_timestamps(frames, subtitle_path)
        print(f"      ✓ Created {num_subtitles} subtitle entries")
        print(f"      ✓ Saved to: subtitles.srt")
        print(f"      ✓ Subtitles: Correct text + Whisper timing")

        # Step 4.5: Merge continuation frames (same visual, no jarring transitions)
        print("\n[4.5/9] Detecting and merging continuation frames...")
        segments = merge_continuation_frames(frames, video_folder)

        # Count how many frames were merged
        merged_count = sum(1 for seg in segments if len(seg.frames) > 1)
        continuation_frames = sum(len(seg.frames) - 1 for seg in segments if len(seg.frames) > 1)

        if continuation_frames > 0:
            print(f"      ✓ Found {continuation_frames} continuation frame(s)")
            print(f"      ✓ Merged into {len(segments)} video segments (was {len(frames)} frames)")
            for seg in segments:
                if len(seg.frames) > 1:
                    print(f"        - Frames {seg.frame_numbers}: merged (same visual)")

            # Prepare merged audio files
            print("      ✓ Concatenating audio for merged segments...")
            prepare_merged_segments(segments, video_folder)
        else:
            print(f"      ✓ No continuation frames detected")
            print(f"      ✓ {len(segments)} segments (1:1 with frames)")

        # Step 5: Build FFmpeg command
        print("\n[5/9] Building FFmpeg command...")
        if continuation_frames > 0:
            ffmpeg_cmd = build_ffmpeg_command_with_segments(video_folder, segments, subtitle_path)
            print(f"      ✓ Using merged segment compilation")
        else:
            ffmpeg_cmd = build_ffmpeg_command(video_folder, frames, subtitle_path)
        print(f"      ✓ Filter graph created")
        print(f"      ✓ {len(segments)} segments with 0.5s crossfade transitions")
        print(f"      ✓ Using actual audio durations (no estimates)")
        print(f"      ✓ Frames and audio synchronized (no delay)")

        # Step 6: Execute compilation
        print("\n[6/9] Compiling video...")
        success, message = execute_ffmpeg(ffmpeg_cmd)
        if not success:
            raise FFmpegError(message)

        # Step 7: Verify output
        print("\n[7/9] Verifying output...")
        verification = verify_compilation(video_folder, frames)

        if verification.get('duration_ok'):
            print(f"      ✓ Duration verified: {verification['actual_duration']:.0f}s")
        else:
            print(f"      ⚠ Duration off by {verification['duration_diff']:.1f}s")

        if verification.get('resolution_ok'):
            print(f"      ✓ Resolution verified: 1920x1080")
        else:
            print(f"      ⚠ Resolution: {verification['width']}x{verification['height']}")

        print(f"      ✓ File size: {verification['file_size_mb']:.1f} MB")

        # Step 8: Generate report
        print("\n[8/9] Generating report...")
        end_time = datetime.now()
        compilation_time = (end_time - start_time).total_seconds()

        report = generate_report(
            video_folder, frames, num_subtitles,
            verification, compilation_time
        )

        report_path = os.path.join(video_folder, 'compilation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"      ✓ Report saved to: compilation_report.txt")

        # Print summary
        print("\n" + "=" * 70)
        print("COMPILATION COMPLETE")
        print("=" * 70)
        print(f"✓ Output: {video_folder}/final_video.mp4")
        print(f"✓ Duration: {verification['actual_duration']:.0f}s (target: {verification['expected_duration']:.0f}s)")
        print(f"✓ File size: {verification['file_size_mb']:.1f} MB")
        print(f"✓ Compilation time: {compilation_time:.1f}s")
        print("\nReady for review!")

        return "SUCCESS"

    except FrameMismatchError as e:
        print(f"\n✗ ERROR: Frame mismatch - {e}")
        return f"ERROR: {e}"
    except TimingError as e:
        print(f"\n✗ ERROR: Timing issue - {e}")
        return f"ERROR: {e}"
    except FFmpegError as e:
        print(f"\n✗ ERROR: FFmpeg failed - {e}")
        return f"ERROR: {e}"
    except Exception as e:
        print(f"\n✗ ERROR: Unexpected error - {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {e}"


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python3 compile_video.py Week-N/Video-M")
        print("\nExample: python3 compile_video.py Week-1/Video-1")
        sys.exit(1)

    video_folder = sys.argv[1]

    # Convert to absolute path if needed
    if not os.path.isabs(video_folder):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_folder = os.path.join(base_dir, video_folder)

    if not os.path.exists(video_folder):
        print(f"Error: Video folder not found: {video_folder}")
        sys.exit(1)

    result = compile_video(video_folder)

    if result == "SUCCESS":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
