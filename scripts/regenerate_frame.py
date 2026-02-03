#!/usr/bin/env python3
"""
Regenerate specific frames with Gemini

Use this script to regenerate individual frames after review.
Allows providing custom instructions to correct issues.

This script uses the SAME prompt structure as generate_slides_gemini.py
to ensure consistency, with the addition of custom correction instructions.

Usage:
    python regenerate_frame.py <video_dir> <frame_number> [--instruction "custom instruction"]

Examples:
    python regenerate_frame.py pipeline/YOUR_LECTURE/Video-2 5
    python regenerate_frame.py pipeline/YOUR_LECTURE/Video-2 5 --instruction "Put option payoff should increase as price decreases below strike"
"""

import sys
import json
import argparse
import io
from pathlib import Path

# Add parent to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.gemini_client import GeminiClient
from scripts.utils.script_parser import load_script
from PIL import Image

# Import shared functions from generate_slides_gemini.py
from scripts.generate_slides_gemini import (
    STYLE_PROMPT,
    MATH_CONTENT_RULES,
    OUTPUT_SIZE,
    is_math_content,
    load_visual_specs,
    load_math_verification,
    get_verified_math_steps,
    get_visual_spec_by_ref,
    classify_frame,
    build_slide_prompt,
    resize_to_1080p,
)


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
            "word_count": frame.word_count,
            "narration": frame.narration,
            "visual_ref": frame.visual.reference if frame.visual else None
        })

    return script_data.title, frames


def regenerate_frame(
    video_dir: Path,
    frame_number: int,
    custom_instruction: str = None,
    verbose: bool = False
) -> bool:
    """
    Regenerate a specific frame with optional custom instruction.

    Uses the same prompt structure as generate_slides_gemini.py for consistency.

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

    visual_ref = frame.get("visual_ref", "") or ""
    narration = frame.get("narration", "")

    # Load visual specs
    specs = load_visual_specs(video_dir)
    requires_math = specs.get("requires_math", False)

    # Load math verification data
    math_data = load_math_verification(video_dir)

    # Get available chart files
    diagrams_dir = video_dir / "diagrams"
    chart_files = [f.name for f in diagrams_dir.glob("*.png")] if diagrams_dir.exists() else []

    # Get visual spec for this frame
    visual_spec = get_visual_spec_by_ref(visual_ref, specs)

    # Classify frame type
    frame_type = classify_frame(frame, visual_spec, chart_files)

    # Get verified math steps if available
    verified_steps = get_verified_math_steps(frame_number, math_data)

    print("=" * 60)
    print(f"Regenerating Frame {frame_number}")
    print("=" * 60)
    print(f"Video: {video_dir}")
    print(f"Title: {title}")
    print(f"Frame type: {frame_type}")
    print(f"Visual: {visual_ref[:60]}..." if len(visual_ref) > 60 else f"Visual: {visual_ref}")
    print(f"Requires math: {requires_math}")
    if verified_steps:
        print(f"Math verification: Available")
    if custom_instruction:
        print(f"Custom instruction: {custom_instruction}")
    print()

    # Build prompt using the SAME function as generate_slides_gemini.py
    prompt = build_slide_prompt(
        frame=frame,
        frame_type=frame_type,
        visual_spec=visual_spec,
        title=title,
        total_frames=len(frames),
        requires_math=requires_math,
        verified_math_steps=verified_steps
    )

    # Add custom instruction if provided
    if custom_instruction:
        prompt += f"""

IMPORTANT CORRECTION/INSTRUCTION:
{custom_instruction}

Make sure to follow this instruction carefully when generating the image.
This correction takes priority over other visual specifications.
"""

    if verbose:
        print("Prompt:")
        print("-" * 40)
        print(prompt)
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
        image_bytes = client.generate_image(
            prompt=prompt,
            style="hand-drawn educational",
            width=1920,
            height=1080
        )
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
