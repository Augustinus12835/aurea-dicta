#!/usr/bin/env python3
"""
TTS Audio Generation Script using ElevenLabs API
Generates individual MP3 files for each frame using ElevenLabs voice synthesis.
Configure VOICE_ID below with your preferred voice (default or cloned).
"""

import os
import sys
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from mutagen.mp3 import MP3
from elevenlabs.client import ElevenLabs

# Load environment variables from project .env file
# Try multiple locations: project root, parent directory
project_root = Path(__file__).parent.parent
env_paths = [
    project_root / '.env',
    Path.home() / '.env',
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break

# API Configuration
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID')
MODEL_ID = "eleven_multilingual_v2"  # High quality model
OUTPUT_FORMAT = "mp3_44100_128"  # 44.1kHz, 128kbps

# Initialize ElevenLabs client
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

class Frame:
    """Represents a single frame with narration"""
    def __init__(self, number, start_time, end_time, word_count, text):
        self.number = number
        self.start_time = start_time
        self.end_time = end_time
        self.word_count = word_count
        self.text = text
        self.duration = self.calculate_duration()

    def calculate_duration(self):
        """Calculate target duration in seconds from time range"""
        start_parts = self.start_time.split(':')
        end_parts = self.end_time.split(':')

        start_seconds = int(start_parts[0]) * 60 + int(start_parts[1])
        end_seconds = int(end_parts[0]) * 60 + int(end_parts[1])

        return end_seconds - start_seconds

    def __repr__(self):
        return f"Frame {self.number}: {self.duration}s, {self.word_count} words"


def parse_script(script_path):
    """
    Parse script.md file to extract frame information

    Expected format:
    ## Frame X (MM:SS-MM:SS) • NN words

    [narration text]
    """
    frames = []

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script file not found: {script_path}")

    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match frame headers
    # ## Frame 0 (0:00-0:15) • 38 words
    pattern = r'## Frame (\d+) \((\d+:\d+)-(\d+:\d+)\) • (\d+) words\n\n(.*?)(?=\n## |\Z)'

    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        frame_num = int(match.group(1))
        start_time = match.group(2)
        end_time = match.group(3)
        word_count = int(match.group(4))
        text = match.group(5).strip()

        # Clean the text - remove markdown formatting
        text = clean_narration_text(text)

        frame = Frame(frame_num, start_time, end_time, word_count, text)
        frames.append(frame)

    return frames


def clean_narration_text(text):
    """Remove markdown formatting, visual annotations, and clean narration text"""
    # Remove [Visual: ...] annotations (match to end since they always appear last)
    text = re.sub(r'\[Visual:.*$', '', text, flags=re.DOTALL)

    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)

    # Remove markdown links [text](url)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)

    # Remove excess whitespace
    text = ' '.join(text.split())

    return text


def call_elevenlabs_api(text):
    """
    Call ElevenLabs API to generate audio from text

    Returns:
        bytes: Audio file content
    """
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY not found in environment variables")

    try:
        # Generate audio using ElevenLabs
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format=OUTPUT_FORMAT
        )

        # Convert generator to bytes
        audio_bytes = b''.join(audio_generator)
        return audio_bytes

    except Exception as e:
        raise Exception(f"ElevenLabs API error: {str(e)}")


def get_audio_duration(file_path):
    """Get duration of MP3 file in seconds"""
    try:
        audio = MP3(file_path)
        return audio.info.length
    except Exception as e:
        print(f"  Warning: Could not read audio duration: {e}")
        return None


def generate_audio_for_frames(frames, output_dir):
    """
    Generate audio files for all frames

    Returns:
        list: Report entries for each frame
    """
    results = []

    # Create audio directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for frame in frames:
        frame_filename = f"frame_{frame.number}.mp3"
        output_path = os.path.join(output_dir, frame_filename)

        print(f"\nProcessing Frame {frame.number}...")
        print(f"  Target duration: {frame.duration}s")
        print(f"  Word count: {frame.word_count}")
        print(f"  Text preview: {frame.text[:60]}...")

        try:
            # Generate audio
            audio_data = call_elevenlabs_api(frame.text)

            # Save to file
            with open(output_path, 'wb') as f:
                f.write(audio_data)

            print(f"  ✓ Saved to {frame_filename}")

            # Verify duration
            actual_duration = get_audio_duration(output_path)

            if actual_duration:
                difference = actual_duration - frame.duration

                result = {
                    'frame': frame.number,
                    'filename': frame_filename,
                    'target': frame.duration,
                    'actual': actual_duration,
                    'difference': difference,
                    'status': 'success'
                }

                # Check if timing is acceptable (within 2 seconds)
                if abs(difference) > 2:
                    result['warning'] = True
                else:
                    result['warning'] = False

                results.append(result)
                print(f"  Duration: {actual_duration:.1f}s (diff: {difference:+.1f}s)")
            else:
                results.append({
                    'frame': frame.number,
                    'filename': frame_filename,
                    'target': frame.duration,
                    'status': 'success',
                    'warning': False,
                    'note': 'Could not verify duration'
                })

        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results.append({
                'frame': frame.number,
                'filename': frame_filename,
                'target': frame.duration,
                'status': 'failed',
                'error': str(e)
            })

        # Small delay between frames to be respectful to the API
        time.sleep(0.5)

    return results


def print_report(results):
    """Print final generation report"""
    print("\n" + "=" * 60)
    print("TTS Generation Complete (ElevenLabs)")
    print("=" * 60)
    print()

    successful = 0
    failed = 0
    needs_adjustment = 0
    total_actual_duration = 0
    total_target_duration = 0

    for result in results:
        if result['status'] == 'success':
            successful += 1

            if 'actual' in result:
                total_actual_duration += result['actual']
                total_target_duration += result['target']

                if result.get('warning', False):
                    needs_adjustment += 1
                    print(f"⚠ {result['filename']}: {result['actual']:.1f}s (target: {result['target']}s) - "
                          f"{abs(result['difference']):.1f}s {'short' if result['difference'] < 0 else 'long'}")
                else:
                    print(f"✓ {result['filename']}: {result['actual']:.1f}s (target: {result['target']}s) - OK")
            else:
                print(f"✓ {result['filename']}: Saved ({result.get('note', '')})")
        else:
            failed += 1
            print(f"✗ {result['filename']}: FAILED - {result['error']}")

    print()
    print("Summary:")
    print(f"- Total frames: {len(results)}")
    print(f"- Successful: {successful}")
    print(f"- Need adjustment: {needs_adjustment}")
    print(f"- Failed: {failed}")

    if total_actual_duration > 0:
        print(f"- Total audio duration: {format_time(total_actual_duration)} "
              f"(target: {format_time(total_target_duration)})")

    print()
    if failed == 0 and needs_adjustment == 0:
        print("✓ All frames generated successfully!")
        print("Next step: Run video compilation")
    elif failed == 0:
        print(f"⚠ Review {needs_adjustment} frame(s) with timing issues")
    else:
        print(f"✗ {failed} frame(s) failed. Review errors above.")


def format_time(seconds):
    """Format seconds as MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python generate_tts_elevenlabs.py <path_to_script.md>")
        print()
        print("Example:")
        print("  python generate_tts_elevenlabs.py Week-1/Video-1/script.md")
        sys.exit(1)

    script_path = sys.argv[1]

    # Determine output directory (same directory as script, in 'audio' subfolder)
    script_dir = os.path.dirname(script_path)
    audio_dir = os.path.join(script_dir, 'audio')

    print("=" * 60)
    print("ElevenLabs TTS Audio Generator")
    print("=" * 60)
    print(f"Script: {script_path}")
    print(f"Output: {audio_dir}")
    print(f"Voice ID: {VOICE_ID}")
    print(f"Model: {MODEL_ID}")
    print("=" * 60)

    # Verify API key and voice ID
    if not ELEVENLABS_API_KEY:
        print("\n✗ Error: ELEVENLABS_API_KEY not found in environment")
        print("  Please check your .env file")
        sys.exit(1)

    if not VOICE_ID:
        print("\n✗ Error: ELEVENLABS_VOICE_ID not found in environment")
        print("  Please add ELEVENLABS_VOICE_ID=your_voice_id to your .env file")
        sys.exit(1)

    try:
        # Parse script
        print("\nParsing script...")
        frames = parse_script(script_path)
        print(f"✓ Found {len(frames)} frames")

        # Generate audio
        results = generate_audio_for_frames(frames, audio_dir)

        # Print report
        print_report(results)

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
