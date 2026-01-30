#!/usr/bin/env python3
"""
Math Verification Script

Verifies mathematical calculations in script using Claude extended thinking.
Generates:
1. Natural English narration for TTS (no symbols)
2. Precise math notation for Gemini slides

Pipeline position:
    script.json → verify_math.py → math_verification.json

Usage:
    python verify_math.py pipeline/YOUR_LECTURE/Video-N
    python verify_math.py pipeline/YOUR_LECTURE/Video-N --frames 5,7,8
    python verify_math.py pipeline/YOUR_LECTURE/Video-N --verbose
    python verify_math.py pipeline/YOUR_LECTURE/Video-N --force
"""

import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add parent to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.claude_client import ClaudeClient
from scripts.utils.script_parser import load_script

# Keywords that indicate a frame needs math verification
MATH_KEYWORDS = [
    # Calculus terms
    'limit', 'derivative', 'integral', 'differentiate', 'differentiation',
    'continuous', 'discontinuous', 'asymptote', 'infinity',
    # Algebraic operations
    'factor', 'factoring', 'simplify', 'simplifies', 'substitute', 'substitution',
    'cancel', 'cancels', 'canceling', 'cancellation',
    'multiply', 'multiplying', 'divide', 'dividing',
    # Expressions and equations
    'equation', 'expression', 'formula', 'calculate', 'calculation',
    'evaluate', 'evaluates', 'equals', 'solve', 'solving',
    # Specific math patterns
    'squared', 'cubed', 'square root', 'root', 'exponent',
    'numerator', 'denominator', 'fraction',
    # Technique markers
    'conjugate', 'difference of squares', 'foil',
    'step', 'steps', 'step-by-step',
    # Results
    'answer', 'result', 'yields', 'gives',
    # Math notation in text
    'x equals', 'x =', 'f(x)', 'lim',
]

# Patterns that strongly indicate calculation content
MATH_PATTERNS = [
    r'x\s*(?:equals?|=)\s*\d',  # x = 2
    r'f\s*\(\s*[a-z]\s*\)',     # f(x)
    r'lim\s*(?:\(|\[)?',        # lim
    r'\d+\s*[+\-*/]\s*\d+',     # 2 + 2
    r'(?:zero|0)\s*/\s*(?:zero|0)',  # 0/0
    r'square\s+root',           # square root
    r'x\s+(?:plus|minus)\s+\d', # x plus 2
]


def parse_script_from_dir(video_dir: Path) -> Tuple[str, List[Dict]]:
    """
    Parse script file (JSON or MD) to extract frame information.

    Uses the shared script_parser utility for consistent parsing.

    Returns:
        (title, frames_list)
    """
    script_data = load_script(video_dir)

    frames = []
    for frame in script_data.frames:
        frames.append({
            "number": frame.number,
            "timing": frame.timing_str,
            "word_count": frame.word_count,
            "narration": frame.narration,
            "visual_ref": frame.visual.reference if frame.visual else ""
        })

    return script_data.title, frames


def load_visual_specs(video_dir: Path) -> Dict:
    """Load visual_specs.json."""
    specs_path = video_dir / "visual_specs.json"
    if not specs_path.exists():
        return {"visuals": [], "requires_math": True}  # Default to True for old videos
    with open(specs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if "requires_math" not in data:
            # Default to True for backwards compatibility (old calculus videos)
            data["requires_math"] = True
        return data


def frame_needs_verification(frame: Dict) -> bool:
    """
    Determine if a frame contains mathematical content needing verification.

    Uses keyword matching and pattern detection to identify frames with
    step-by-step calculations.
    """
    narration = frame.get("narration", "").lower()
    visual_ref = frame.get("visual_ref", "").lower()
    combined = f"{narration} {visual_ref}"

    # Count keyword matches
    keyword_count = sum(1 for kw in MATH_KEYWORDS if kw in combined)

    # Check for math patterns
    pattern_matches = any(re.search(p, combined) for p in MATH_PATTERNS)

    # Strong indicators of calculation content
    has_walkthrough = any(w in combined for w in ['walkthrough', 'step-by-step', 'step 1', 'step 2'])
    has_substitution = 'substitut' in combined
    has_cancellation = 'cancel' in combined

    # Decision logic
    if has_walkthrough or has_substitution or has_cancellation:
        return True
    if keyword_count >= 4:
        return True
    if keyword_count >= 2 and pattern_matches:
        return True

    return False


def verify_frame(client: ClaudeClient, frame: Dict, verbose: bool = False) -> Dict:
    """
    Verify a single frame's mathematical content.

    Returns verification result with natural narration and math steps.
    """
    frame_num = frame["number"]
    narration = frame["narration"]
    visual_ref = frame.get("visual_ref", "")

    if verbose:
        print(f"\n    Verifying frame {frame_num}...")
        print(f"    Narration preview: {narration[:100]}...")

    result = client.verify_math(
        narration=narration,
        visual_context=visual_ref,
        frame_number=frame_num,
        budget_tokens=10000
    )

    if verbose:
        status = result.get("verification_status", "unknown")
        confidence = result.get("confidence", "unknown")
        print(f"    Status: {status}, Confidence: {confidence}")
        if result.get("issues_found"):
            print(f"    Issues: {result['issues_found']}")

    return result


def run_verification(
    video_dir: Path,
    specific_frames: List[int] = None,
    verbose: bool = False,
    force: bool = False
) -> Dict:
    """
    Run math verification for a video.

    Args:
        video_dir: Path to Video-N directory
        specific_frames: Optional list of frame numbers to verify
        verbose: Show detailed output
        force: Re-verify even if math_verification.json exists

    Returns:
        Verification results dict
    """
    print(f"\n{'='*60}")
    print(f"Math Verification: {video_dir.name}")
    print(f"{'='*60}")

    # Check for existing verification
    output_path = video_dir / "math_verification.json"
    if output_path.exists() and not force and not specific_frames:
        print(f"  math_verification.json already exists. Use --force to regenerate.")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Load inputs - check for script.json or script.md
    json_path = video_dir / "script.json"
    md_path = video_dir / "script.md"
    if not json_path.exists() and not md_path.exists():
        return {"success": False, "error": "No script.json or script.md found"}

    specs = load_visual_specs(video_dir)
    requires_math = specs.get("requires_math", True)

    if not requires_math:
        print(f"  Skipping: requires_math=false in visual_specs.json")
        return {"success": True, "skipped": True, "reason": "requires_math=false"}

    title, frames = parse_script_from_dir(video_dir)

    print(f"  Title: {title}")
    print(f"  Total frames: {len(frames)}")

    # Determine which frames to verify
    if specific_frames:
        frames_to_verify = [f for f in frames if f["number"] in specific_frames]
    else:
        frames_to_verify = [f for f in frames if frame_needs_verification(f)]

    print(f"  Frames needing verification: {len(frames_to_verify)}")
    if frames_to_verify:
        print(f"  Frame numbers: {[f['number'] for f in frames_to_verify]}")

    if not frames_to_verify:
        print(f"  No frames require math verification")
        result = {
            "success": True,
            "video_title": title,
            "requires_math": requires_math,
            "frames": {}
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    # Initialize Claude client
    print(f"\n  Initializing Claude client with extended thinking...")
    client = ClaudeClient()

    # Verify each frame
    results = {
        "success": True,
        "video_title": title,
        "requires_math": requires_math,
        "frames": {}
    }

    verified_count = 0
    error_count = 0

    for frame in frames_to_verify:
        frame_num = frame["number"]
        print(f"\n  Frame {frame_num}:")

        try:
            verification = verify_frame(client, frame, verbose)

            results["frames"][str(frame_num)] = {
                "requires_verification": True,
                "verification_status": verification.get("verification_status", "unknown"),
                "natural_narration": verification.get("natural_narration", frame["narration"]),
                "math_steps": verification.get("math_steps", []),
                "final_answer": verification.get("final_answer"),
                "issues_found": verification.get("issues_found", []),
                "confidence": verification.get("confidence", "unknown"),
                "original_narration": frame["narration"]
            }

            status = verification.get("verification_status", "unknown")
            if status == "correct":
                print(f"    ✓ Verified: {status}")
                verified_count += 1
            elif status == "corrected":
                print(f"    ⚠ Corrections made")
                verified_count += 1
            elif status == "error":
                print(f"    ✗ Error: {verification.get('error', 'unknown')}")
                error_count += 1
            else:
                print(f"    ? Status: {status}")
                verified_count += 1

        except Exception as e:
            print(f"    ✗ Error: {e}")
            results["frames"][str(frame_num)] = {
                "requires_verification": True,
                "verification_status": "error",
                "error": str(e),
                "original_narration": frame["narration"]
            }
            error_count += 1

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Summary:")
    print(f"    Verified: {verified_count}")
    print(f"    Errors: {error_count}")
    print(f"    Output: {output_path}")

    if error_count > 0:
        results["success"] = False

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify mathematical content in video scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python verify_math.py pipeline/YOUR_LECTURE/Video-1
    python verify_math.py pipeline/YOUR_LECTURE/Video-1 --frames 5,7,8
    python verify_math.py pipeline/YOUR_LECTURE/Video-1 --verbose --force
        """
    )

    parser.add_argument("video_dir", help="Path to Video-N directory")
    parser.add_argument("--frames", help="Comma-separated list of frame numbers to verify")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-verification")

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: Directory not found: {video_dir}")
        sys.exit(1)

    specific_frames = None
    if args.frames:
        specific_frames = [int(f.strip()) for f in args.frames.split(",")]

    print("="*60)
    print("Math Verification (Claude Extended Thinking)")
    print("="*60)
    print(f"Video: {video_dir}")
    if specific_frames:
        print(f"Frames: {specific_frames}")
    if args.force:
        print("Mode: Force re-verification")

    result = run_verification(
        video_dir,
        specific_frames=specific_frames,
        verbose=args.verbose,
        force=args.force
    )

    print("\n" + "="*60)
    if result.get("success"):
        if result.get("skipped"):
            print("VERIFICATION SKIPPED (not a math video)")
        else:
            print("VERIFICATION COMPLETE")
            print(f"\nOutput: {video_dir}/math_verification.json")
            print("\nNext steps:")
            print(f"  1. Review math_verification.json for accuracy")
            print(f"  2. Run frame generation: python scripts/generate_slides_gemini.py {video_dir}")
            print(f"  3. Run TTS: python scripts/generate_tts_elevenlabs.py {video_dir}/script.md")
    else:
        print("VERIFICATION FAILED")
        print(f"\nErrors occurred. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
