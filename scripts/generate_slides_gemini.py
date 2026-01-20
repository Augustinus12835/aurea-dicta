#!/usr/bin/env python3
"""
Slide Generation using Gemini 2.0 Flash

Generates slide PNGs directly from script + visual specs.
Fully automated slide generation.

Pipeline position:
  script.md + visual_specs.json + diagrams/*.png
      → generate_slides_gemini.py
      → slides/*.png (ready for video compilation)

For data charts: Composites pre-generated charts onto slide backgrounds.
For conceptual diagrams: Generates from visual_specs.json specifications.

Usage:
    python generate_slides_gemini.py pipeline/YOUR_LECTURE/Video-1
    python generate_slides_gemini.py pipeline/YOUR_LECTURE --all-videos
"""

import os
import sys
import re
import json
import time
import io
import base64
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add parent to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import PIL for image compositing
try:
    from PIL import Image
except ImportError:
    print("Error: PIL/Pillow required. Install with: pip install Pillow")
    sys.exit(1)

from scripts.utils.gemini_client import GeminiClient

# Configuration
OUTPUT_SIZE = (1920, 1080)

# Style constants for prompts
STYLE_PROMPT = """Educational slide with hand-drawn, sketch-like aesthetic.

STYLE REQUIREMENTS:
- Hand-drawn illustration style (NOT corporate PowerPoint)
- Clean white or cream background (#FFFEF7)
- Color palette:
  * Blue (#3B82F6) for primary concepts and headers
  * Orange (#F97316) for highlights and emphasis
  * Green (#22C55E) for positive outcomes and success
  * Red (#EF4444) for warnings and cautions
- Typography:
  * Hand-written style headers (bold, clear)
  * Clean sans-serif body text (legible at 1080p)
- Professional but approachable
- NO course codes, university branding, or dates
- Resolution: 1920x1080 (16:9)
"""


def parse_script(script_path: Path) -> Tuple[str, List[Dict]]:
    """
    Parse script.md to extract frame information.

    Returns:
        (title, frames_list)

    frames_list structure:
    [
        {
            "number": 0,
            "timing": "0:00-0:30",
            "word_count": 75,
            "narration": "Full narration text...",
            "visual_ref": "visual_7.png" or "Conceptual - Risk vs Uncertainty Matrix"
        },
        ...
    ]
    """
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract title
    title_match = re.search(r"# Script: (.+)", content)
    title = title_match.group(1).strip() if title_match else "Untitled"

    # Parse frames - match: ## Frame N (timing) • X words
    frames = []

    # Split by frame headers
    frame_pattern = r"## Frame (\d+) \(([^)]+)\)(?: • (\d+) words?)?"
    sections = re.split(r"(?=## Frame \d+)", content)

    for section in sections:
        if not section.strip() or not section.strip().startswith("## Frame"):
            continue

        # Extract header info
        header_match = re.search(frame_pattern, section)
        if not header_match:
            continue

        frame_num = int(header_match.group(1))
        timing = header_match.group(2)
        word_count = int(header_match.group(3)) if header_match.group(3) else 0

        # Extract visual reference
        visual_match = re.search(r"\[Visual: ([^\]]+)\]", section)
        visual_ref = visual_match.group(1).strip() if visual_match else None

        # Extract narration (text between header and [Visual:])
        # Remove header line and visual line
        lines = section.split("\n")
        narration_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("## Frame"):
                continue
            if line.startswith("[Visual:"):
                continue
            if line.startswith("---"):
                continue
            narration_lines.append(line)

        narration = " ".join(narration_lines).strip()

        if not word_count:
            word_count = len(narration.split())

        frames.append({
            "number": frame_num,
            "timing": timing,
            "word_count": word_count,
            "narration": narration,
            "visual_ref": visual_ref
        })

    return title, frames


def load_visual_specs(video_dir: Path) -> Dict:
    """Load visual_specs.json."""
    specs_path = video_dir / "visual_specs.json"
    if not specs_path.exists():
        return {"visuals": []}
    with open(specs_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_visual_spec_by_ref(visual_ref: str, specs: Dict) -> Optional[Dict]:
    """Find visual spec matching a reference."""
    if not visual_ref:
        return None

    # Check for PNG reference (e.g., "visual_7.png")
    if ".png" in visual_ref:
        visual_id = visual_ref.replace(".png", "").strip()
        for v in specs.get("visuals", []):
            if v.get("id") == visual_id:
                return v

    # Check for name match in conceptual reference
    for v in specs.get("visuals", []):
        if v.get("name", "").lower() in visual_ref.lower():
            return v

    return None


def classify_frame(frame: Dict, visual_spec: Optional[Dict], chart_files: List[str]) -> str:
    """
    Classify frame for generation strategy.

    Returns:
        "data_chart"      → Composite generation
        "conceptual"      → Full Gemini generation
        "text_focused"    → Text-heavy slide
        "title"           → Title/closing slide
    """
    visual_ref = frame.get("visual_ref", "") or ""
    frame_num = frame.get("number", 0)

    # Title slide (Frame 0 usually)
    if frame_num == 0:
        return "title"

    # Check if it references a data chart PNG (e.g., "visual_7.png - description...")
    if visual_ref and ".png" in visual_ref:
        # Extract just the filename before any description
        if " - " in visual_ref:
            chart_filename = visual_ref.split(" - ")[0].strip()
        else:
            chart_filename = visual_ref.strip()

        # Ensure we have just the .png filename
        if not chart_filename.endswith(".png"):
            chart_filename = chart_filename.split(".png")[0] + ".png"

        # Check if this chart file exists in our available charts
        if chart_filename in chart_files:
            return "data_chart"

    # Check visual spec type
    if visual_spec:
        spec_type = visual_spec.get("type", "")
        if spec_type == "data_chart":
            return "data_chart"
        elif spec_type in ("conceptual_diagram", "comparison_table", "matrix", "flowchart"):
            return "conceptual"

    # Check for conceptual references
    if visual_ref and "conceptual" in visual_ref.lower():
        return "conceptual"

    # Default to text-focused
    return "text_focused"


def build_conceptual_diagram_prompt(visual_spec: Optional[Dict]) -> str:
    """
    Build detailed prompt for conceptual diagram from visual_specs.json.
    """
    if not visual_spec:
        return "Generate a relevant conceptual diagram based on the narration."

    name = visual_spec.get("name", "Diagram")
    purpose = visual_spec.get("purpose", "")
    elements = visual_spec.get("elements", [])
    style = visual_spec.get("style", "hand-drawn")

    prompt = f"""
DIAGRAM: {name}
PURPOSE: {purpose}
STYLE: {style}

ELEMENTS TO INCLUDE:
"""

    for elem in elements:
        if isinstance(elem, dict):
            elem_type = elem.get("type", "")

            if elem_type == "matrix":
                prompt += f"- 2x2 Matrix with {elem.get('rows', 2)} rows and {elem.get('cols', 2)} columns\n"

            elif elem_type == "axis_label":
                position = elem.get("position", "")
                text = elem.get("text", "")
                values = elem.get("values", [])
                prompt += f"- {position.upper()} axis: '{text}' with values {values}\n"

            elif elem_type == "quadrant":
                position = elem.get("position", "")
                label = elem.get("label", "")
                color = elem.get("color", "")
                examples = elem.get("examples", [])
                prompt += f"- {position.upper()} quadrant: '{label}' ({color})\n"
                if examples:
                    prompt += f"    Examples: {', '.join(str(e) for e in examples[:3])}\n"

            elif elem_type in ("node", "decision"):
                label = elem.get("label", "")
                shape = elem.get("shape", "rectangle")
                color = elem.get("color", "")
                prompt += f"- {elem_type.title()}: '{label}' ({shape}, {color})\n"

            elif elem_type == "branch":
                label = elem.get("label", "")
                prob = elem.get("probability", "")
                prompt += f"- Branch: '{label}' {f'({prob})' if prob else ''}\n"

            elif elem_type == "calculation":
                formula = elem.get("formula", "")
                prompt += f"- Show calculation: {formula}\n"

            elif elem_type == "annotation":
                text = elem.get("text", "")
                prompt += f"- Annotation: '{text}'\n"

            else:
                # Generic element
                label = elem.get("label", elem.get("text", str(elem)))
                prompt += f"- {label}\n"
        else:
            prompt += f"- {elem}\n"

    return prompt


def build_slide_prompt(
    frame: Dict,
    frame_type: str,
    visual_spec: Optional[Dict],
    title: str,
    total_frames: int
) -> str:
    """
    Build Gemini prompt for slide generation.
    """
    frame_num = frame["number"]
    narration = frame["narration"]
    timing = frame["timing"]

    # Truncate narration if too long
    narration_preview = narration[:500] if len(narration) > 500 else narration

    base_prompt = f"""{STYLE_PROMPT}

VIDEO CONTEXT:
- Topic: "{title}"

NARRATION FOR THIS FRAME:
"{narration_preview}"

CRITICAL: DO NOT include any frame numbers, slide numbers, timestamps,
durations, "Frame X of Y", or any technical metadata anywhere in the
generated image. The image should contain ONLY the visual content.
"""

    if frame_type == "title":
        return base_prompt + """
SLIDE TYPE: Title Slide

REQUIREMENTS:
- Large, bold title centered or top-third
- Subtitle or key question below (from narration)
- Small preview element or icon related to the topic
- Clean, professional, inviting
- NO course codes or dates
- Hand-drawn visual element (dice, graph sketch, question mark, etc.)
"""

    elif frame_type == "data_chart":
        chart_name = visual_spec.get("name", "Chart") if visual_spec else "Data Chart"
        chart_purpose = visual_spec.get("purpose", "") if visual_spec else ""

        return base_prompt + f"""
SLIDE TYPE: Data Chart Integration

THIS SLIDE WILL HAVE A DATA CHART OVERLAID.
Generate a slide background/layout that:
- Has a clear title area at top: "{chart_name}"
- Has a large central area (60-70% of slide) that will hold the chart
- LEAVE THE CENTRAL AREA MOSTLY EMPTY (light background, no conflicting elements)
- Add hand-drawn callout arrows or annotation placeholders around the edges
- Include brief explanatory text or key insight at bottom if space allows

PURPOSE OF CHART: {chart_purpose}

The actual chart PNG will be composited onto this layout.
Focus on the FRAME/LAYOUT only - do not draw the chart itself.
"""

    elif frame_type == "conceptual":
        # Build detailed diagram instructions from visual_spec
        diagram_prompt = build_conceptual_diagram_prompt(visual_spec)

        return base_prompt + f"""
SLIDE TYPE: Conceptual Diagram

GENERATE THIS DIAGRAM:
{diagram_prompt}

REQUIREMENTS:
- Hand-drawn, sketch-like style
- Clear visual hierarchy
- Legible labels and text
- Use color coding: Blue (primary), Orange (highlight), Green (positive), Red (warning)
- Simple enough to understand at a glance
- Include arrows, connections, or flow indicators where appropriate
"""

    else:  # text_focused
        return base_prompt + f"""
SLIDE TYPE: Text-Focused Content

REQUIREMENTS:
- Extract 3-5 key points from the narration
- Large, readable bullet points or numbered list
- Hand-drawn icons or simple illustrations to support each point
- NOT just a wall of text - visual interest required
- Generous white space
- Clear visual hierarchy
- Use checkmarks, arrows, or hand-drawn numbering
"""


def resize_to_1080p(image_bytes: bytes) -> bytes:
    """
    Resize image to exactly 1920x1080 (1080p).

    Args:
        image_bytes: Input image as bytes

    Returns:
        Resized image as PNG bytes
    """
    img = Image.open(io.BytesIO(image_bytes))

    if img.size != OUTPUT_SIZE:
        img = img.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)

    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def generate_slide_with_gemini(
    client: GeminiClient,
    prompt: str,
    max_retries: int = 3
) -> Optional[bytes]:
    """
    Generate slide using Gemini with retry logic.

    Returns:
        PNG image bytes (resized to 1920x1080) or None on failure
    """
    for attempt in range(max_retries):
        try:
            image_bytes = client.generate_image(
                prompt=prompt,
                style="hand-drawn educational",
                width=1920,
                height=1080
            )
            # Resize to exact 1080p
            return resize_to_1080p(image_bytes)

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"      Retry {attempt + 1}/{max_retries} in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"      Failed after {max_retries} attempts: {e}")
                return None

    return None


def composite_chart_onto_slide(
    slide_bytes: bytes,
    chart_path: Path,
    position: str = "center"
) -> bytes:
    """
    Composite data chart PNG onto generated slide background.

    Args:
        slide_bytes: Generated slide background as bytes
        chart_path: Path to chart PNG
        position: "center", "left", "right", "top", "bottom"

    Returns:
        Composited image as PNG bytes
    """
    # Load slide background
    slide = Image.open(io.BytesIO(slide_bytes))

    # Ensure slide is right size
    if slide.size != OUTPUT_SIZE:
        slide = slide.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)

    # Convert to RGBA if needed
    if slide.mode != "RGBA":
        slide = slide.convert("RGBA")

    # Load chart
    chart = Image.open(chart_path)
    if chart.mode != "RGBA":
        chart = chart.convert("RGBA")

    # Calculate target size (chart should be ~65% of slide width)
    target_width = int(OUTPUT_SIZE[0] * 0.65)
    target_height = int(OUTPUT_SIZE[1] * 0.65)

    # Maintain aspect ratio
    chart_ratio = chart.width / chart.height
    target_ratio = target_width / target_height

    if chart_ratio > target_ratio:
        # Chart is wider - fit to width
        new_width = target_width
        new_height = int(target_width / chart_ratio)
    else:
        # Chart is taller - fit to height
        new_height = target_height
        new_width = int(target_height * chart_ratio)

    chart = chart.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Calculate position
    if position == "center":
        x = (OUTPUT_SIZE[0] - new_width) // 2
        y = (OUTPUT_SIZE[1] - new_height) // 2 + 30  # Slight offset for title
    elif position == "left":
        x = 50
        y = (OUTPUT_SIZE[1] - new_height) // 2
    elif position == "right":
        x = OUTPUT_SIZE[0] - new_width - 50
        y = (OUTPUT_SIZE[1] - new_height) // 2
    else:
        x = (OUTPUT_SIZE[0] - new_width) // 2
        y = (OUTPUT_SIZE[1] - new_height) // 2

    # Composite
    slide.paste(chart, (x, y), chart)

    # Convert back to bytes
    output = io.BytesIO()
    slide.save(output, format="PNG")
    return output.getvalue()


def validate_slide(slide_path: Path) -> Dict:
    """Validate generated slide meets requirements."""
    if not slide_path.exists():
        return {"valid": False, "error": "File does not exist"}

    try:
        img = Image.open(slide_path)
        size_ok = img.size == OUTPUT_SIZE

        return {
            "valid": size_ok,
            "resolution_ok": size_ok,
            "format_ok": slide_path.suffix == ".png",
            "file_size_kb": slide_path.stat().st_size / 1024,
            "actual_size": img.size,
            "mode": img.mode
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def copy_and_resize_chart(chart_path: Path, output_path: Path) -> bytes:
    """
    Copy data chart and resize to 1920x1080.

    Args:
        chart_path: Source chart PNG
        output_path: Destination path

    Returns:
        PNG bytes
    """
    img = Image.open(chart_path)

    # Resize to 1080p
    if img.size != OUTPUT_SIZE:
        img = img.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)

    # Save
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


def generate_frames_for_video(video_dir: Path, verbose: bool = False, specific_frame: int = None) -> Dict:
    """
    Generate all frames for a video.

    For data charts: copies existing PNG from diagrams/ folder
    For conceptual diagrams: generates with Gemini

    Output goes to frames/ folder (ready for video compilation).

    Returns:
        Result dict with success/failure info
    """
    print(f"\n{'='*60}")
    print(f"Generating frames: {video_dir.name}")
    print(f"{'='*60}")

    # Load inputs
    script_path = video_dir / "script.md"
    if not script_path.exists():
        return {"success": False, "error": "No script.md found"}

    title, frames = parse_script(script_path)
    specs = load_visual_specs(video_dir)

    # Get available chart files
    diagrams_dir = video_dir / "diagrams"
    chart_files = [f.name for f in diagrams_dir.glob("*.png")] if diagrams_dir.exists() else []

    print(f"  Title: {title}")
    print(f"  Frames: {len(frames)}")
    print(f"  Data charts available: {len(chart_files)}")
    print(f"  Visual specs: {len(specs.get('visuals', []))}")

    # Filter to specific frame if requested
    if specific_frame is not None:
        frames = [f for f in frames if f["number"] == specific_frame]
        if not frames:
            return {"success": False, "error": f"Frame {specific_frame} not found"}
        print(f"  Processing frame {specific_frame} only")

    # Create output directory - frames/ (not slides/)
    frames_dir = video_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Track results
    results = {
        "success": True,
        "frames_generated": [],
        "frames_copied": [],
        "frames_failed": [],
        "metadata": {
            "title": title,
            "total_frames": len(frames),
            "frames": []
        }
    }

    # Initialize Gemini client (only if needed)
    client = None

    for frame in frames:
        frame_num = frame["number"]
        visual_ref = frame.get("visual_ref", "")

        print(f"\n  Frame {frame_num}:")

        # Get visual spec for this frame
        visual_spec = get_visual_spec_by_ref(visual_ref, specs)

        # Classify frame
        frame_type = classify_frame(frame, visual_spec, chart_files)
        print(f"    Type: {frame_type}")

        if visual_ref:
            print(f"    Visual: {visual_ref[:60]}...")

        output_path = frames_dir / f"frame_{frame_num}.png"

        try:
            # For data charts: copy existing PNG from diagrams/
            if frame_type == "data_chart" and visual_ref and ".png" in visual_ref:
                # Extract just the filename (e.g., "visual_7.png" from "visual_7.png - description...")
                chart_filename = visual_ref.split(" - ")[0].strip()
                if not chart_filename.endswith(".png"):
                    chart_filename = chart_filename.split(".png")[0] + ".png"
                chart_id = chart_filename.replace(".png", "")
                chart_path = diagrams_dir / f"{chart_id}.png"

                if chart_path.exists():
                    print(f"    Copying chart: {chart_id}.png → frame_{frame_num}.png")
                    frame_bytes = copy_and_resize_chart(chart_path, output_path)

                    with open(output_path, "wb") as f:
                        f.write(frame_bytes)

                    results["frames_copied"].append(frame_num)
                    print(f"    Copied and resized to 1920x1080")
                else:
                    print(f"    ERROR: Chart not found: {chart_path}")
                    results["frames_failed"].append(frame_num)
                    results["success"] = False
                    continue

            else:
                # For conceptual/title/text frames: generate with Gemini
                if client is None:
                    print(f"    Initializing Gemini client...")
                    try:
                        client = GeminiClient()
                        print(f"    Using model: {client.model}")
                    except Exception as e:
                        print(f"    ERROR: Gemini client init failed: {e}")
                        results["frames_failed"].append(frame_num)
                        results["success"] = False
                        continue

                # Build prompt
                prompt = build_slide_prompt(
                    frame, frame_type, visual_spec, title, len(frames)
                )

                if verbose:
                    print(f"    Prompt preview: {prompt[:300]}...")

                print(f"    Generating with Gemini...")
                frame_bytes = generate_slide_with_gemini(client, prompt)

                if not frame_bytes:
                    print(f"    ERROR: Failed to generate frame")
                    results["frames_failed"].append(frame_num)
                    results["success"] = False
                    continue

                with open(output_path, "wb") as f:
                    f.write(frame_bytes)

                results["frames_generated"].append(frame_num)

            # Validate
            validation = validate_slide(output_path)
            if validation["valid"]:
                print(f"    Saved: {output_path.name} ({validation['file_size_kb']:.1f} KB)")
            else:
                print(f"    Warning: Validation issue - {validation}")

            results["metadata"]["frames"].append({
                "number": frame_num,
                "timing": frame["timing"],
                "type": frame_type,
                "visual_ref": visual_ref,
                "file": f"frame_{frame_num}.png"
            })

        except Exception as e:
            print(f"    ERROR: {e}")
            results["frames_failed"].append(frame_num)
            results["success"] = False

    # Save metadata
    metadata_path = frames_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(results["metadata"], f, indent=2)

    print(f"\n  Summary:")
    print(f"    Generated (Gemini): {len(results['frames_generated'])}")
    print(f"    Copied (data charts): {len(results['frames_copied'])}")
    print(f"    Failed: {len(results['frames_failed'])}")

    return results


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_slides_gemini.py <video_dir> [--verbose] [--frame N]")
        print("       python generate_slides_gemini.py <pipeline_dir> --all-videos [--verbose]")
        print()
        print("Generates frame PNGs for video compilation.")
        print("  - Data charts: copied from diagrams/ folder")
        print("  - Conceptual diagrams: generated with Gemini")
        print()
        print("Options:")
        print("  --all-videos  Process all Video-* directories")
        print("  --verbose     Show detailed output")
        print("  --frame N     Generate specific frame only (for testing)")
        print()
        print("Examples:")
        print("  python generate_slides_gemini.py pipeline/YOUR_LECTURE/Video-1")
        print("  python generate_slides_gemini.py pipeline/YOUR_LECTURE/Video-1 --frame 0")
        print("  python generate_slides_gemini.py pipeline/YOUR_LECTURE --all-videos")
        sys.exit(1)

    target_path = Path(sys.argv[1])
    verbose = "--verbose" in sys.argv
    all_videos = "--all-videos" in sys.argv

    specific_frame = None
    if "--frame" in sys.argv:
        idx = sys.argv.index("--frame")
        if idx + 1 < len(sys.argv):
            specific_frame = int(sys.argv[idx + 1])

    print("=" * 60)
    print("Frame Generation (Gemini + Data Charts)")
    print("=" * 60)
    print(f"Target: {target_path}")
    if specific_frame is not None:
        print(f"Frame: {specific_frame} only")

    if all_videos:
        # Process all Video-* directories
        video_dirs = sorted(target_path.glob("Video-*"))
        if not video_dirs:
            print(f"No Video-* directories found in {target_path}")
            sys.exit(1)

        all_results = {"success": [], "failed": []}

        for video_dir in video_dirs:
            result = generate_frames_for_video(video_dir, verbose, specific_frame)
            video_name = video_dir.name

            if result.get("success"):
                gen = len(result.get('frames_generated', []))
                copied = len(result.get('frames_copied', []))
                all_results["success"].append(f"{video_name}: {gen} generated, {copied} copied")
            else:
                all_results["failed"].append(f"{video_name}: {result.get('error', 'unknown error')}")

        print("\n" + "=" * 60)
        print("ALL VIDEOS COMPLETE")
        print("=" * 60)

        if all_results["success"]:
            print("\nSuccess:")
            for s in all_results["success"]:
                print(f"  - {s}")

        if all_results["failed"]:
            print("\nFailed:")
            for f in all_results["failed"]:
                print(f"  - {f}")
    else:
        # Process single video
        if not target_path.is_dir():
            print(f"Error: {target_path} is not a directory")
            sys.exit(1)

        result = generate_frames_for_video(target_path, verbose, specific_frame)

        print("\n" + "=" * 60)
        print("FRAME GENERATION COMPLETE")
        print("=" * 60)

        if result.get("success"):
            gen = len(result.get('frames_generated', []))
            copied = len(result.get('frames_copied', []))
            print(f"\nGenerated {gen} frames, copied {copied} data charts")
            print(f"Output: {target_path}/frames/")
            print(f"\nNext step: python scripts/generate_tts_elevenlabs.py {target_path}/script.md")
        else:
            print(f"\nGeneration failed: {result.get('error', 'Unknown error')}")
            if result.get("frames_failed"):
                print(f"Failed frames: {result['frames_failed']}")


if __name__ == "__main__":
    main()
