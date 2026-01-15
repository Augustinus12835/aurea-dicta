#!/usr/bin/env python3
"""
Script Generation for Aurea Dicta

Generates script.md with frame-by-frame narration.
Uses video_brief.md and references actual chart files created.

Pipeline position:
  video_brief.md + visual_specs.json + diagrams/*.png
      → generate_scripts.py
      → script.md (frame-by-frame narration with visual references)
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.claude_client import ClaudeClient


SCRIPT_GENERATION_PROMPT = """You are writing a narration script for an educational video.

VIDEO BRIEF:
{video_brief}

AVAILABLE DATA CHARTS (PNG files that will be shown):
{chart_list}

CONCEPTUAL DIAGRAMS (Gemini will create these):
{conceptual_list}

TEACHING STYLE GUIDE:
{style_guide}

---

Create a frame-by-frame narration script following this EXACT format:

## Frame 0 (0:00-0:XX) • NN words

[Opening narration - corresponds to title slide]

[Visual: Title slide]

---

## Frame 1 (0:XX-Y:YY) • NN words

[Narration for this frame]

[Visual: visual_1.png - description] OR [Visual: Conceptual diagram description]

---

[Continue for all frames...]

REQUIREMENTS:

1. **Timing:**
   - Target 2.5 words per second
   - Total duration should match video_brief estimate (typically 5-7 minutes)
   - Each frame: 15-60 seconds typically

2. **Frame Count:**
   - Typically 8-12 frames for a 6-8 minute video
   - Frame 0 = Title/Hook
   - Last Frame = Synthesis/Closing

3. **Visual References:**
   - When a frame uses a data chart, note: [Visual: chart_id.png - brief description]
   - When a frame needs a conceptual diagram, note: [Visual: Conceptual - description of diagram]
   - Match visuals to teaching flow from brief
   - Reference ALL available data charts at appropriate moments

4. **Script Style:**
   - Concise conversational (per teaching_style_guide)
   - No filler words ("um", "basically", "actually", "kind of")
   - Active voice throughout
   - Define technical terms on first use
   - No rhetorical questions or verbal cushioning

5. **Teaching Flow Alignment:**
   - Hook frames first (engaging opening)
   - Build frames (establish concepts)
   - Deepen frames (nuance, common confusions)
   - Apply frames (practical application)
   - Follow the structure in video_brief

6. **Visual Distinctiveness (CRITICAL):**
   - Each frame MUST have a UNIQUE, STANDALONE visual concept
   - NEVER use "(continued)" or "continued focus" for any visual
   - If explaining a complex concept over multiple frames:
     * Frame A: Show one aspect/perspective of the concept
     * Frame B: Show a DIFFERENT aspect or zoom into specific detail
     * Frame C: Show practical application or real-world example
   - Each [Visual:] description must be self-contained and distinct
   - BAD: "[Visual: Decision Framework (continued)]"
   - BAD: "[Visual: Matrix (continued focus on strategic implications)]"
   - GOOD: "[Visual: Decision Framework - Risk Assessment Branch]"
   - GOOD: "[Visual: Matrix showing Risk quadrant with hedging strategies]"
   - GOOD: "[Visual: Matrix showing Uncertainty quadrant with buffer strategies]"
   - When reusing a concept, show a DIFFERENT visual angle or specific subset

OUTPUT FORMAT:
```markdown
# Script: [Video Title]

**Total Duration:** X:XX
**Frame Count:** N
**Word Count:** NNN (target: N words at 2.5/sec)

---

## Frame 0 (0:00-0:20) • 50 words

[Narration text...]

[Visual: Title slide with key concept preview]

---

## Frame 1 (0:20-0:50) • 75 words

[Narration text...]

[Visual: visual_1.png - VIX chart showing fear spikes]

---

## Frame 2 (0:50-1:25) • 88 words

[Narration text...]

[Visual: Conceptual - Long vs Short position profit diagrams]

---
```

Ensure word counts match timing (seconds × 2.5).
Reference each available data chart at least once.

IMPORTANT: The narration should end naturally with the final teaching point. Do NOT include any meta-commentary, summary statistics, or "that's all for today" style endings in the spoken narration.
"""


def load_video_brief(video_dir: Path) -> str:
    """Load video_brief.md content."""
    brief_path = video_dir / "video_brief.md"
    if not brief_path.exists():
        raise FileNotFoundError(f"video_brief.md not found in {video_dir}")
    with open(brief_path, "r", encoding="utf-8") as f:
        return f.read()


def load_visual_specs(video_dir: Path) -> Dict:
    """Load visual_specs.json."""
    specs_path = video_dir / "visual_specs.json"
    if not specs_path.exists():
        return {"visuals": []}
    with open(specs_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_created_charts(video_dir: Path) -> List[str]:
    """Get list of created chart PNG files."""
    diagrams_dir = video_dir / "diagrams"
    if not diagrams_dir.exists():
        return []
    return sorted([f.name for f in diagrams_dir.glob("*.png")])


def load_style_guide() -> Optional[str]:
    """Load teaching style guide if available."""
    style_paths = [
        Path(__file__).parent.parent / "templates" / "teaching_style_guide.md",
        Path(__file__).parent.parent / "teaching_style_guide.md",
    ]

    for path in style_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # Return condensed version for prompt (key points only)
                return extract_style_key_points(content)

    return "Use concise conversational style. No filler words. Active voice. 2.5 words/second pacing."


def extract_style_key_points(_style_guide: str) -> str:
    """Extract key points from style guide to keep prompt concise."""
    # Return condensed version focusing on actionable guidelines
    # (Full style guide is too long for prompt; using curated key points)
    return """Key Style Points:
- Concise conversational: Every word earns its place
- NO filler words: "um", "basically", "actually", "kind of", "you know"
- NO rhetorical questions: State facts directly
- NO verbal cushioning: "So basically what I'm trying to say is..."
- Active voice always
- Contractions OK but sparingly
- 2.5 words per second pacing
- Define technical terms on first use (one sentence)
- State each concept ONCE, don't repeat
- "We" only when doing something together
- "You" only when action required"""


def format_chart_list(specs: Dict, created_charts: List[str]) -> str:
    """Format list of data charts for prompt."""
    data_charts = [v for v in specs.get("visuals", []) if v.get("type") == "data_chart"]

    if not data_charts:
        return "None"

    lines = []
    for chart in data_charts:
        chart_id = chart.get("id", "unknown")
        name = chart.get("name", "Unnamed")
        purpose = chart.get("purpose", "")
        section = chart.get("teaching_section", "")
        filename = f"{chart_id}.png"

        status = "AVAILABLE" if filename in created_charts else "NOT CREATED"
        lines.append(f"- **{filename}**: {name}")
        lines.append(f"  Purpose: {purpose}")
        lines.append(f"  Teaching section: {section}")
        lines.append(f"  Status: {status}")

    return "\n".join(lines)


def format_conceptual_list(specs: Dict) -> str:
    """Format list of conceptual diagrams for prompt."""
    conceptual = [v for v in specs.get("visuals", []) if v.get("type") in ("conceptual_diagram", "comparison_table")]

    if not conceptual:
        return "None"

    lines = []
    for diagram in conceptual:
        name = diagram.get("name", "Unnamed")
        purpose = diagram.get("purpose", "")
        section = diagram.get("teaching_section", "")
        elements = diagram.get("elements", [])

        lines.append(f"- **{name}**")
        lines.append(f"  Purpose: {purpose}")
        lines.append(f"  Teaching section: {section}")
        if elements:
            # Show first few elements as preview
            if isinstance(elements, list):
                # Elements might be strings or dicts
                preview = []
                for elem in elements[:3]:
                    if isinstance(elem, str):
                        preview.append(elem[:50])
                    elif isinstance(elem, dict):
                        # Extract key info from dict
                        preview.append(str(elem.get("label", elem.get("type", str(elem)[:30]))))
                    else:
                        preview.append(str(elem)[:30])
                elem_str = "; ".join(preview)
                if len(elements) > 3:
                    elem_str += f"; ... ({len(elements)} total)"
            else:
                elem_str = str(elements)[:150]
            lines.append(f"  Key elements: {elem_str}")

    return "\n".join(lines)


def generate_script(
    client: ClaudeClient,
    video_brief: str,
    chart_list: str,
    conceptual_list: str,
    style_guide: str
) -> str:
    """Generate script using Claude."""

    prompt = SCRIPT_GENERATION_PROMPT.format(
        video_brief=video_brief,
        chart_list=chart_list,
        conceptual_list=conceptual_list,
        style_guide=style_guide
    )

    response = client.generate(
        prompt=prompt,
        system="You are an expert educational script writer. Create clear, engaging narration that follows the teaching flow precisely. Match visuals to content. Be concise - no filler words.",
        max_tokens=6000,
        temperature=0.5
    )

    return response


def count_script_stats(script: str) -> Dict:
    """Count frames, words, and estimate duration."""
    import re

    frame_count = len(re.findall(r"## Frame \d+", script))
    word_count = len(script.split())

    # Try to extract duration from script header
    duration_match = re.search(r"\*\*Total Duration:\*\*\s*(\d+:\d+)", script)
    duration = duration_match.group(1) if duration_match else f"~{word_count // 150}:00"

    return {
        "frames": frame_count,
        "words": word_count,
        "duration": duration
    }


def main():
    """Main script generation function."""
    if len(sys.argv) < 2:
        print("Usage: python generate_scripts.py <pipeline_dir> [--video N] [--verbose]")
        print()
        print("Generates narration scripts from video briefs with visual references.")
        print()
        print("Options:")
        print("  --video N     Generate script for specific video only")
        print("  --verbose     Show detailed output")
        print()
        print("Examples:")
        print("  python generate_scripts.py pipeline/YOUR_LECTURE")
        print("  python generate_scripts.py pipeline/YOUR_LECTURE --video 2")
        sys.exit(1)

    pipeline_dir = sys.argv[1]
    specific_video = None
    verbose = "--verbose" in sys.argv

    if "--video" in sys.argv:
        idx = sys.argv.index("--video")
        if idx + 1 < len(sys.argv):
            specific_video = int(sys.argv[idx + 1])

    print("=" * 60)
    print("Script Generation")
    print("=" * 60)
    print(f"Pipeline: {pipeline_dir}")
    if specific_video:
        print(f"Video: {specific_video}")

    pipeline_path = Path(pipeline_dir)

    # Find video directories
    if specific_video:
        video_dirs = [pipeline_path / f"Video-{specific_video}"]
    else:
        video_dirs = sorted(pipeline_path.glob("Video-*"))

    if not video_dirs:
        print(f"No video directories found in {pipeline_dir}")
        sys.exit(1)

    # Load style guide once
    style_guide = load_style_guide()
    print(f"Style guide: {'Loaded' if style_guide else 'Using defaults'}")

    # Initialize Claude client
    client = ClaudeClient()

    results = {"success": [], "failed": []}

    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue

        video_name = video_dir.name
        print(f"\n{video_name}:")

        # Load video brief
        try:
            video_brief = load_video_brief(video_dir)
            print(f"  Loaded: video_brief.md")
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            results["failed"].append(video_name)
            continue

        # Load visual specs
        specs = load_visual_specs(video_dir)

        # Get created charts
        created_charts = get_created_charts(video_dir)
        print(f"  Data charts available: {len(created_charts)}")
        if verbose and created_charts:
            for chart in created_charts:
                print(f"    - {chart}")

        # Count conceptual diagrams
        conceptual_count = len([v for v in specs.get("visuals", [])
                               if v.get("type") in ("conceptual_diagram", "comparison_table")])
        print(f"  Conceptual diagrams: {conceptual_count} (for Gemini)")

        # Format lists for prompt
        chart_list = format_chart_list(specs, created_charts)
        conceptual_list = format_conceptual_list(specs)

        # Generate script
        print(f"  Generating script...")
        try:
            script = generate_script(client, video_brief, chart_list, conceptual_list, style_guide)
        except Exception as e:
            print(f"  Error generating script: {e}")
            results["failed"].append(video_name)
            continue

        # Save script
        script_path = video_dir / "script.md"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        # Count stats
        stats = count_script_stats(script)

        print(f"  Saved: {script_path}")
        print(f"    Frames: {stats['frames']}")
        print(f"    Words: ~{stats['words']}")
        print(f"    Duration: {stats['duration']}")

        results["success"].append(video_name)

    # Summary
    print("\n" + "=" * 60)
    print("SCRIPT GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Success: {len(results['success'])}")
    print(f"  Failed:  {len(results['failed'])}")

    if results["failed"]:
        print("\nFailed videos:")
        for v in results["failed"]:
            print(f"  - {v}")

    if results["success"]:
        print(f"\nNext step: python scripts/generate_slide_instructions.py {pipeline_dir}")


if __name__ == "__main__":
    main()
