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
- Color usage (for consistency):
  * Blue (#3B82F6) for primary concepts and headers
  * Orange (#F97316) for highlights and emphasis
  * Green (#22C55E) for positive outcomes
  * Red (#EF4444) for warnings or negative outcomes
- Typography:
  * Hand-written style headers (bold, clear)
  * Minimal text - labels only, NOT sentences or paragraphs
- Professional but approachable
- NO course codes, university branding, or dates
- NO color legends or keys on the slide
- Resolution: 1920x1080 (16:9)

CRITICAL TEXT RULES:
- This slide COMPLEMENTS spoken narration - the viewer will HEAR the explanation
- DO NOT put sentences, paragraphs, or bullet points of text on the slide
- Text should be LIMITED TO: titles, axis labels, short annotations (1-5 words max)
- If you need to show a concept, DRAW IT, don't write about it
- The slide should be 80% visual, 20% text (labels only)
"""

# Math-specific prompt additions
MATH_CONTENT_RULES = """
MATH CONTENT RULES (this slide contains mathematical content):
- Show the CALCULATION PROCESS step-by-step using mathematical notation
- Use proper math symbols: fractions, exponents, integrals, limits, Greek letters
- The slide should be REPRODUCIBLE - a student can follow the math just from the slide
- Show: formulas → substitution → intermediate steps → final answer
- Use arrows (→) or equals signs to show progression between steps
- Box or highlight the final answer
- Label key values and variables
- Graphs should have labeled axes with scale/units
- OK to have more text IF it's mathematical notation (equations, formulas, expressions)
- Still avoid prose/sentences - use math symbols instead of words where possible
"""

# No-math prompt additions (for non-quantitative content)
NO_MATH_RULES = """
CRITICAL - NO MATHEMATICAL NOTATION:
This video does NOT require mathematical representations. DO NOT use:
- Mathematical formulas, equations, or expressions
- Set notation (∈, ⊂, {}, ∪, ∩)
- Function notation (f(x), g(x))
- Greek letters used mathematically (Σ, π, θ as variables)
- Proofs, derivations, or "therefore" (∴) symbols
- Subscripts/superscripts for variables (C₁, C₂, S_T)
- Mathematical inequalities as logical statements

Instead, use:
- Simple text labels and annotations (1-5 words)
- Visual diagrams with arrows and connections
- Maps, timelines, comparison tables
- Icons and illustrations
- Plain language descriptions
"""


def is_math_content(narration: str, visual_ref: str, title: str) -> bool:
    """
    Detect if the content is mathematical and needs step-by-step notation.
    """
    math_keywords = [
        'calculus', 'derivative', 'integral', 'limit', 'function',
        'equation', 'formula', 'solve', 'calculate', 'compute',
        'graph', 'slope', 'tangent', 'algebra', 'polynomial',
        'factor', 'simplify', 'evaluate', 'substitute', 'expression',
        'x =', 'y =', 'f(x)', 'f(', 'equals', 'plus', 'minus',
        'squared', 'cubed', 'root', 'exponent', 'logarithm',
        'sin', 'cos', 'tan', 'theorem', 'proof', 'delta',
        'continuous', 'discontinuous', 'asymptote', 'infinity'
    ]

    combined_text = f"{narration} {visual_ref} {title}".lower()

    # Check for math keywords
    keyword_matches = sum(1 for kw in math_keywords if kw in combined_text)

    # Check for mathematical patterns
    import re
    has_math_notation = bool(re.search(r'[a-zA-Z]\s*[=<>≤≥]\s*\d|f\s*\(|lim|∫|Σ|π|θ|\d+/\d+', combined_text))

    return keyword_matches >= 2 or has_math_notation


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
        return {"visuals": [], "requires_math": False}  # Default: no math
    with open(specs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Ensure requires_math exists (backward compatibility)
        if "requires_math" not in data:
            data["requires_math"] = False  # Default: no math (safer)
        return data


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
    Focuses on VISUAL elements, not text.
    """
    if not visual_spec:
        return "Generate a simple visual diagram. Use icons and shapes, minimal text."

    name = visual_spec.get("name", "Diagram")
    purpose = visual_spec.get("purpose", "")
    elements = visual_spec.get("elements", [])
    style = visual_spec.get("style", "hand-drawn")

    prompt = f"""
DIAGRAM NAME: {name}
VISUAL GOAL: {purpose}

DRAW THESE VISUAL ELEMENTS (use shapes/icons, label with 1-3 words only):
"""

    for elem in elements:
        if isinstance(elem, dict):
            elem_type = elem.get("type", "")

            if elem_type == "matrix":
                prompt += f"- Draw a {elem.get('rows', 2)}x{elem.get('cols', 2)} grid/matrix\n"

            elif elem_type == "axis_label":
                position = elem.get("position", "")
                text = elem.get("text", "")
                prompt += f"- {position.upper()} axis labeled '{text}'\n"

            elif elem_type == "quadrant":
                position = elem.get("position", "")
                label = elem.get("label", "")
                color = elem.get("color", "")
                prompt += f"- {position} section: label '{label}' ({color})\n"

            elif elem_type in ("node", "decision"):
                label = elem.get("label", "")
                shape = elem.get("shape", "rectangle")
                prompt += f"- Draw {shape} labeled '{label}'\n"

            elif elem_type == "branch":
                label = elem.get("label", "")
                prompt += f"- Arrow/branch: '{label}'\n"

            elif elem_type == "calculation":
                formula = elem.get("formula", "")
                prompt += f"- Show formula: {formula}\n"

            elif elem_type == "annotation":
                text = elem.get("text", "")
                # Only include if it's short
                if len(text) < 30:
                    prompt += f"- Small label: '{text}'\n"

            else:
                # Generic element - extract just the visual part
                label = elem.get("label", elem.get("text", str(elem)))
                if len(str(label)) < 50:  # Skip long text elements
                    prompt += f"- {label}\n"
        else:
            # String elements - only include if visual/short
            if len(str(elem)) < 50 and not any(word in str(elem).lower() for word in ['explain', 'describe', 'narration']):
                prompt += f"- {elem}\n"

    prompt += """
REMEMBER: Labels should be 1-5 words MAX. Draw the concept, don't write about it.
"""
    return prompt


def extract_key_concepts(narration: str) -> str:
    """
    Extract key concepts from narration for visual guidance.
    Returns a brief summary of what to visualize, not the full text.
    """
    # Extract key terms (words in quotes, technical terms, numbers)
    import re

    # Find quoted terms
    quoted = re.findall(r'"([^"]+)"', narration)

    # Find mathematical expressions (simple patterns)
    math_terms = re.findall(r'[a-zA-Z]\s*(?:of\s+)?[a-zA-Z]\s*(?:equals?|=)\s*\d+', narration)

    # Find numbers with context
    numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:percent|%|dollars?|\$))?\b', narration)

    # Get first sentence as topic indicator (truncated)
    first_sentence = narration.split('.')[0][:100] if narration else ""

    concepts = []
    if first_sentence:
        concepts.append(f"Topic: {first_sentence}")
    if quoted:
        concepts.append(f"Key terms: {', '.join(quoted[:3])}")
    if numbers:
        concepts.append(f"Key values: {', '.join(numbers[:3])}")

    return "\n".join(concepts) if concepts else "General educational content"


def build_slide_prompt(
    frame: Dict,
    frame_type: str,
    visual_spec: Optional[Dict],
    title: str,
    total_frames: int,
    requires_math: bool = False
) -> str:
    """
    Build Gemini prompt for slide generation.

    Args:
        frame: Frame dict with number, narration, visual_ref
        frame_type: "title", "data_chart", "conceptual", or "text_focused"
        visual_spec: Visual specification dict from visual_specs.json
        title: Video title
        total_frames: Total number of frames in video
        requires_math: Whether this video requires mathematical notation
    """
    frame_num = frame["number"]
    narration = frame["narration"]
    timing = frame["timing"]
    visual_ref = frame.get("visual_ref", "")

    # Determine if this specific frame needs math content
    # Only check for math keywords if the video allows math
    if requires_math:
        is_math = is_math_content(narration, visual_ref, title)
    else:
        is_math = False

    # Extract key concepts instead of full narration
    key_concepts = extract_key_concepts(narration)

    # For math content, also extract formulas and expressions
    if is_math:
        import re
        # Find formula-like patterns to include
        formulas = re.findall(r'[a-zA-Z]+\s*\([^)]+\)\s*=\s*[^,.]+|[a-zA-Z]\s*=\s*[^,.]+|\d+\s*[+\-*/]\s*\d+', narration)
        if formulas:
            key_concepts += f"\nKey formulas/expressions: {'; '.join(formulas[:5])}"

    # Build base prompt with appropriate math rules
    if requires_math:
        math_section = MATH_CONTENT_RULES if is_math else ""
    else:
        math_section = NO_MATH_RULES  # Explicit no-math rules for non-quantitative content

    base_prompt = f"""{STYLE_PROMPT}
{math_section}
VIDEO CONTEXT:
- Topic: "{title}"
- Content type: {"MATHEMATICAL/COMPUTATIONAL" if is_math else "CONCEPTUAL"}

KEY CONCEPTS FOR THIS FRAME:
{key_concepts}

VISUAL DESCRIPTION (from script):
{visual_ref if visual_ref else "Create an appropriate visual for the topic"}

CRITICAL RULES:
- DO NOT include frame numbers, slide numbers, timestamps, or metadata
- DO NOT write sentences or paragraphs - use SHORT LABELS only (1-5 words)
- DO NOT repeat the narration as text - the viewer will HEAR it
- FOCUS on diagrams, graphs, icons, and visual representations
"""

    if frame_type == "title":
        return base_prompt + """
SLIDE TYPE: Title Slide

REQUIREMENTS:
- Large, bold title centered or top-third
- ONE key visual element that represents the topic (icon, simple diagram, or metaphor)
- Minimal text: just the title (5-10 words max)
- NO subtitle paragraphs - let the narration introduce the topic
- Hand-drawn aesthetic
- The visual element should hint at what's coming (curiosity hook)
"""

    elif frame_type == "data_chart":
        chart_name = visual_spec.get("name", "Chart") if visual_spec else "Data Chart"
        chart_purpose = visual_spec.get("purpose", "") if visual_spec else ""

        return base_prompt + f"""
SLIDE TYPE: Data Chart Integration

THIS SLIDE WILL HAVE A DATA CHART OVERLAID.
Generate a slide background/layout that:
- Has a SHORT title at top (3-5 words): based on "{chart_name}"
- Has a large central area (70% of slide) for the chart - LEAVE EMPTY
- Optional: 1-2 hand-drawn annotation arrows pointing to where key insights will be
- NO explanatory text - the narration will explain the chart

The actual chart PNG will be composited onto this layout.
"""

    elif frame_type == "conceptual":
        # Build detailed diagram instructions from visual_spec
        diagram_prompt = build_conceptual_diagram_prompt(visual_spec)

        if is_math:
            return base_prompt + f"""
SLIDE TYPE: Mathematical Diagram/Derivation

GENERATE THIS VISUAL:
{diagram_prompt}

MATH-SPECIFIC REQUIREMENTS:
- Show the mathematical process STEP BY STEP
- Use proper mathematical notation (fractions, exponents, symbols)
- Include: starting formula → substitution → simplification → answer
- Label each step or use arrows (→) to show progression
- Box or highlight the FINAL ANSWER
- Graphs must have labeled axes with values
- A student should be able to reproduce the calculation from this slide alone
- OK to have mathematical expressions - avoid prose/sentences
"""
        else:
            return base_prompt + f"""
SLIDE TYPE: Conceptual Diagram

GENERATE THIS VISUAL:
{diagram_prompt}

REQUIREMENTS:
- Hand-drawn, sketch-like style
- PRIMARILY VISUAL: diagrams, flowcharts, graphs, icons
- Text limited to SHORT LABELS only (1-5 words per label)
- NO sentences or explanations - the narration provides that
- Use color to differentiate: Blue=primary, Orange=emphasis, Green=positive, Red=negative
- Clear visual hierarchy - main concept should be obvious at a glance
- Arrows and connections to show relationships
"""

    else:  # text_focused
        if is_math:
            return base_prompt + f"""
SLIDE TYPE: Mathematical Summary/Process

REQUIREMENTS:
- Show the KEY FORMULAS or CALCULATION STEPS
- Use mathematical notation, not prose
- Structure as: Given → Formula → Steps → Result
- Include important values and their meanings
- Graphs with labeled axes if applicable
- Box the key formula or final answer
- A student should understand the math process from this slide
- Avoid sentences - use symbols, equations, and short labels
"""
        else:
            return base_prompt + f"""
SLIDE TYPE: Visual Summary (NOT text-focused!)

IMPORTANT: Even though this frame doesn't have a specific diagram, it should still be VISUAL.

REQUIREMENTS:
- Create a VISUAL REPRESENTATION of the concept, not bullet points
- Options for visual approaches:
  * Simple diagram or flowchart showing the process/concept
  * Icons with SHORT labels (1-3 words each)
  * Visual metaphor or analogy illustration
  * Comparison visual (side-by-side, before/after)
  * Timeline or sequence illustration
- Maximum 5-10 words of text TOTAL on the entire slide
- NO bullet points, NO sentences, NO paragraphs
- The narration will explain everything - the slide just needs to SHOW it
- Think: "What would a whiteboard sketch look like?"
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


def generate_frames_for_video(video_dir: Path, verbose: bool = False, specific_frame: int = None, force: bool = False, fail_fast: bool = True) -> Dict:
    """
    Generate all frames for a video.

    For data charts: copies existing PNG from diagrams/ folder
    For conceptual diagrams: generates with Gemini

    Output goes to frames/ folder (ready for video compilation).

    Args:
        video_dir: Path to Video-N directory
        verbose: Show detailed output
        specific_frame: Generate only this frame number (None = all)
        force: Regenerate all frames even if they exist
        fail_fast: Stop on first error (default True)

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

    # Get requires_math flag from specs
    requires_math = specs.get("requires_math", False)

    print(f"  Title: {title}")
    print(f"  Frames: {len(frames)}")
    print(f"  Data charts available: {len(chart_files)}")
    print(f"  Visual specs: {len(specs.get('visuals', []))}")
    print(f"  Requires math: {requires_math}")

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
        "frames_continued": [],  # Frames that copy previous (same visual)
        "frames_skipped": [],
        "frames_failed": [],
        "metadata": {
            "title": title,
            "total_frames": len(frames),
            "frames": []
        }
    }

    # Initialize Gemini client (only if needed)
    client = None

    # Track previous frame's visual spec for continuation detection
    prev_visual_spec_id = None
    prev_frame_num = None

    for frame in frames:
        frame_num = frame["number"]
        visual_ref = frame.get("visual_ref", "")

        output_path = frames_dir / f"frame_{frame_num}.png"

        # Skip if frame already exists (unless --force or regenerating specific frame)
        if not force and specific_frame is None and output_path.exists():
            # Still add to metadata for tracking
            visual_spec = get_visual_spec_by_ref(visual_ref, specs)
            frame_type = classify_frame(frame, visual_spec, chart_files)
            results["metadata"]["frames"].append({
                "number": frame_num,
                "timing": frame["timing"],
                "type": frame_type,
                "visual_ref": visual_ref,
                "file": f"frame_{frame_num}.png"
            })
            results["frames_skipped"].append(frame_num)
            print(f"\n  Frame {frame_num}: exists, skipping")
            continue

        print(f"\n  Frame {frame_num}:")

        # Get visual spec for this frame
        visual_spec = get_visual_spec_by_ref(visual_ref, specs)
        current_visual_spec_id = visual_spec.get("id") if visual_spec else None

        # Classify frame
        frame_type = classify_frame(frame, visual_spec, chart_files)
        print(f"    Type: {frame_type}")

        if visual_ref:
            print(f"    Visual: {visual_ref[:60]}...")

        try:
            # Check for continuation: same visual spec as previous frame
            # This avoids regenerating nearly identical images and enables smooth transitions
            if (current_visual_spec_id and
                current_visual_spec_id == prev_visual_spec_id and
                prev_frame_num is not None and
                frame_type not in ("title", "data_chart")):  # Don't continue title or data chart frames

                prev_frame_path = frames_dir / f"frame_{prev_frame_num}.png"
                if prev_frame_path.exists():
                    import shutil
                    shutil.copy(prev_frame_path, output_path)
                    print(f"    Continuation of frame {prev_frame_num} (same visual spec: {current_visual_spec_id})")
                    print(f"    Copied: frame_{prev_frame_num}.png → frame_{frame_num}.png")

                    results["frames_continued"].append(frame_num)

                    # Add to metadata with continuation marker
                    results["metadata"]["frames"].append({
                        "number": frame_num,
                        "timing": frame["timing"],
                        "type": frame_type,
                        "visual_ref": visual_ref,
                        "file": f"frame_{frame_num}.png",
                        "continuation_of": prev_frame_num,
                        "visual_spec_id": current_visual_spec_id
                    })

                    # Update tracking for next iteration
                    prev_visual_spec_id = current_visual_spec_id
                    prev_frame_num = frame_num
                    continue
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
                    if fail_fast:
                        break
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
                        if fail_fast:
                            break
                        continue

                # Build prompt
                prompt = build_slide_prompt(
                    frame, frame_type, visual_spec, title, len(frames),
                    requires_math=requires_math
                )

                if verbose:
                    print(f"    Prompt preview: {prompt[:300]}...")

                print(f"    Generating with Gemini...")
                frame_bytes = generate_slide_with_gemini(client, prompt)

                if not frame_bytes:
                    print(f"    ERROR: Failed to generate frame")
                    results["frames_failed"].append(frame_num)
                    results["success"] = False
                    if fail_fast:
                        break
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
                "file": f"frame_{frame_num}.png",
                "visual_spec_id": current_visual_spec_id
            })

            # Update tracking for continuation detection
            prev_visual_spec_id = current_visual_spec_id
            prev_frame_num = frame_num

        except Exception as e:
            print(f"    ERROR: {e}")
            results["frames_failed"].append(frame_num)
            results["success"] = False
            if fail_fast:
                break

    # Save metadata
    metadata_path = frames_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(results["metadata"], f, indent=2)

    print(f"\n  Summary:")
    print(f"    Generated (Gemini): {len(results['frames_generated'])}")
    print(f"    Copied (data charts): {len(results['frames_copied'])}")
    print(f"    Continued (same visual): {len(results['frames_continued'])}")
    print(f"    Skipped (existing): {len(results['frames_skipped'])}")
    print(f"    Failed: {len(results['frames_failed'])}")

    return results


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_slides_gemini.py <video_dir> [--verbose] [--frame N] [--force]")
        print("       python generate_slides_gemini.py <pipeline_dir> --all-videos [--verbose] [--force]")
        print()
        print("Generates frame PNGs for video compilation.")
        print("  - Data charts: copied from diagrams/ folder")
        print("  - Conceptual diagrams: generated with Gemini")
        print()
        print("Options:")
        print("  --all-videos          Process all Video-* directories")
        print("  --verbose             Show detailed output")
        print("  --frame N             Generate specific frame only (for testing)")
        print("  --force               Regenerate all frames (ignore existing)")
        print("  --continue-on-error   Don't stop on first error (try all frames)")
        print()
        print("Examples:")
        print("  python generate_slides_gemini.py pipeline/YOUR_LECTURE/Video-1")
        print("  python generate_slides_gemini.py pipeline/YOUR_LECTURE/Video-1 --frame 0")
        print("  python generate_slides_gemini.py pipeline/YOUR_LECTURE --all-videos")
        print("  python generate_slides_gemini.py pipeline/YOUR_LECTURE/Video-1 --force")
        sys.exit(1)

    target_path = Path(sys.argv[1])
    verbose = "--verbose" in sys.argv
    all_videos = "--all-videos" in sys.argv
    force = "--force" in sys.argv
    fail_fast = "--continue-on-error" not in sys.argv

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
    if force:
        print("Mode: Force regenerate all frames")

    if all_videos:
        # Process all Video-* directories
        video_dirs = sorted(target_path.glob("Video-*"))
        if not video_dirs:
            print(f"No Video-* directories found in {target_path}")
            sys.exit(1)

        all_results = {"success": [], "failed": []}

        for video_dir in video_dirs:
            result = generate_frames_for_video(video_dir, verbose, specific_frame, force, fail_fast)
            video_name = video_dir.name

            if result.get("success"):
                gen = len(result.get('frames_generated', []))
                copied = len(result.get('frames_copied', []))
                skipped = len(result.get('frames_skipped', []))
                all_results["success"].append(f"{video_name}: {gen} generated, {copied} copied, {skipped} skipped")
            else:
                all_results["failed"].append(f"{video_name}: {result.get('error', 'unknown error')}")
                if fail_fast:
                    break

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
            sys.exit(1)
    else:
        # Process single video
        if not target_path.is_dir():
            print(f"Error: {target_path} is not a directory")
            sys.exit(1)

        result = generate_frames_for_video(target_path, verbose, specific_frame, force, fail_fast)

        print("\n" + "=" * 60)
        print("FRAME GENERATION COMPLETE")
        print("=" * 60)

        if result.get("success"):
            gen = len(result.get('frames_generated', []))
            copied = len(result.get('frames_copied', []))
            skipped = len(result.get('frames_skipped', []))
            print(f"\nGenerated {gen} frames, copied {copied} data charts, skipped {skipped} existing")
            print(f"Output: {target_path}/frames/")
            print(f"\nNext step: python scripts/generate_tts_elevenlabs.py {target_path}/script.md")
        else:
            print(f"\nGeneration failed: {result.get('error', 'Unknown error')}")
            if result.get("frames_failed"):
                print(f"Failed frames: {result['frames_failed']}")
            print("\nTip: Check the error message above for details.")
            print("     Common issues: content policy, rate limiting, or prompt complexity.")
            print("     Try simplifying the visual description in video_brief.md or visual_specs.json")
            sys.exit(1)


if __name__ == "__main__":
    main()
