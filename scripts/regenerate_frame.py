#!/usr/bin/env python3
"""
Regenerate specific frames with Gemini

Use this script to regenerate individual frames after review.
Allows providing custom instructions to correct issues.

Usage:
    python regenerate_frame.py <video_dir> <frame_number> [--instruction "custom instruction"]

Examples:
    python regenerate_frame.py pipeline/YOUR_LECTURE/Video-2 5
    python regenerate_frame.py pipeline/YOUR_LECTURE/Video-2 5 --instruction "Put option payoff should increase as price decreases below strike"
"""

import sys
import json
import argparse
from pathlib import Path
import io

# Add parent to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.gemini_client import GeminiClient
from scripts.utils.script_parser import load_script
from PIL import Image

# Configuration
OUTPUT_SIZE = (1920, 1080)

# Style prompt (same as generate_slides_gemini.py)
STYLE_PROMPT = """Educational slide with hand-drawn, sketch-like aesthetic.

STYLE REQUIREMENTS:
- Hand-drawn illustration style (NOT corporate PowerPoint)
- Clean white or cream background (#FFFEF7)
- Color palette:
  * Blue (#3B82F6) for risk-related concepts
  * Orange (#F97316) for return-related concepts
  * Green (#22C55E) for time/growth concepts
  * Red (#EF4444) for warnings/uncertainty
- Typography:
  * Hand-written style headers (bold, clear)
  * Clean sans-serif body text (legible at 1080p)
- Professional but approachable
- NO course codes, university branding, or dates
- Resolution: 1920x1080 (16:9)
"""

# No-math content: we simply omit all math-related prompt text entirely.
# By not mentioning math at all, we avoid activating math-related neural pathways.


def load_visual_specs(video_dir: Path) -> dict:
    """Load visual_specs.json."""
    import json
    specs_path = video_dir / "visual_specs.json"
    if not specs_path.exists():
        return {"visuals": [], "requires_math": False}  # Default: no math
    with open(specs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Ensure requires_math exists (backward compatibility)
        if "requires_math" not in data:
            data["requires_math"] = False  # Default: no math (safer)
        return data


def parse_script_from_dir(video_dir: Path) -> tuple:
    """
    Parse script file (JSON or MD) to extract title and frames.

    Uses the shared script_parser utility for consistent parsing.
    """
    script_data = load_script(video_dir)

    frames = []
    for frame in script_data.frames:
        frames.append({
            "number": frame.number,
            "timing": frame.timing_str,
            "narration": frame.narration,
            "visual_ref": frame.visual.reference if frame.visual else None
        })

    return script_data.title, frames


def resize_to_1080p(image_bytes: bytes) -> bytes:
    """Resize image to exactly 1920x1080."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.size != OUTPUT_SIZE:
        img = img.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def regenerate_frame(
    video_dir: Path,
    frame_number: int,
    custom_instruction: str = None,
    verbose: bool = False
) -> bool:
    """
    Regenerate a specific frame with optional custom instruction.

    Args:
        video_dir: Path to video directory
        frame_number: Frame number to regenerate
        custom_instruction: Optional instruction to improve/correct the generation
        verbose: Print detailed output

    Returns:
        True if successful
    """
    frames_dir = video_dir / "frames"

    # Check for script.json or script.md
    json_path = video_dir / "script.json"
    md_path = video_dir / "script.md"
    if not json_path.exists() and not md_path.exists():
        print(f"Error: No script.json or script.md found in {video_dir}")
        return False

    # Parse script
    title, frames = parse_script_from_dir(video_dir)

    # Find the specific frame
    frame = None
    for f in frames:
        if f["number"] == frame_number:
            frame = f
            break

    if frame is None:
        print(f"Error: Frame {frame_number} not found in script")
        return False

    visual_ref = frame.get("visual_ref", "")
    narration = frame.get("narration", "")

    # Load visual specs to get requires_math flag
    specs = load_visual_specs(video_dir)
    requires_math = specs.get("requires_math", False)

    print("=" * 60)
    print(f"Regenerating Frame {frame_number}")
    print("=" * 60)
    print(f"Video: {video_dir}")
    print(f"Visual: {visual_ref[:60]}..." if len(visual_ref) > 60 else f"Visual: {visual_ref}")
    print(f"Requires math: {requires_math}")
    if custom_instruction:
        print(f"Custom instruction: {custom_instruction}")
    print()

    # For non-math content: include NO math-related text at all
    # This avoids activating math-related neural pathways in the model
    narration_preview = narration[:500] if len(narration) > 500 else narration

    prompt = f"""{STYLE_PROMPT}

VIDEO CONTEXT:
- Topic: "{title}"

NARRATION FOR THIS FRAME:
"{narration_preview}"

VISUAL SPECIFICATION:
{visual_ref}

CRITICAL: DO NOT include any frame numbers, slide numbers, timestamps,
durations, "Frame X of Y", or any technical metadata anywhere in the
generated image. The image should contain ONLY the visual content.
"""

    # Add custom instruction if provided
    if custom_instruction:
        prompt += f"""

IMPORTANT CORRECTION/INSTRUCTION:
{custom_instruction}

Make sure to follow this instruction carefully when generating the image.
"""

    if verbose:
        print("Prompt:")
        print("-" * 40)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("-" * 40)
        print()

    # Initialize Gemini and generate
    print("Initializing Gemini client...")
    try:
        client = GeminiClient()
        print(f"Using model: {client.model}")
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        return False

    print("Generating image...")
    try:
        image_bytes = client.generate_image(prompt)
        image_bytes = resize_to_1080p(image_bytes)
    except Exception as e:
        print(f"Error generating image: {e}")
        return False

    # Save
    output_path = frames_dir / f"frame_{frame_number}.png"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(image_bytes)

    file_size = len(image_bytes) / 1024
    print(f"Saved: {output_path} ({file_size:.1f} KB)")
    print()
    print("Done! Please review the regenerated frame.")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate specific frames with Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python regenerate_frame.py pipeline/YOUR_LECTURE/Video-2 5
  python regenerate_frame.py pipeline/YOUR_LECTURE/Video-2 5 --instruction "Put payoff increases as price falls"
  python regenerate_frame.py pipeline/YOUR_LECTURE/Video-2 5 -i "Show strike price at $50, payoff rising linearly below strike"
        """
    )
    parser.add_argument("video_dir", help="Path to video directory")
    parser.add_argument("frame_number", type=int, help="Frame number to regenerate")
    parser.add_argument(
        "-i", "--instruction",
        help="Custom instruction to correct/improve the generation"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output including full prompt"
    )

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: Directory not found: {video_dir}")
        sys.exit(1)

    success = regenerate_frame(
        video_dir,
        args.frame_number,
        custom_instruction=args.instruction,
        verbose=args.verbose
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
