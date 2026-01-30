#!/usr/bin/env python3
"""
Script Generation for Aurea Dicta

Generates script.json (structured data) and script.md (human-readable) with
frame-by-frame narration. Uses video_brief.md and references actual chart files.

Pipeline position:
  video_brief.md + visual_specs.json + diagrams/*.png
      → generate_scripts.py
      → script.json (source of truth, structured data)
      → script.md (derived from JSON, for human review)
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.claude_client import ClaudeClient
from scripts.utils.script_parser import (
    ScriptData, ScriptMetadata, Frame, VisualInfo,
    parse_time_to_seconds, seconds_to_time_str,
    script_to_markdown, script_to_json, save_script
)


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

Create a frame-by-frame narration script as a JSON object.

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
   - When a frame uses a data chart, set type to "data_chart" and reference to "chart_id.png - brief description"
   - When a frame needs a conceptual diagram, set type to "conceptual" and reference to the description
   - Frame 0 should have type "title"
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
   - Each visual reference must be self-contained and distinct
   - BAD: "Decision Framework (continued)"
   - GOOD: "Decision Framework - Risk Assessment Branch"
   - When reusing a concept, show a DIFFERENT visual angle or specific subset

7. **Narration Content:**
   - The "narration" field contains ONLY the spoken text
   - Do NOT include visual annotations or descriptions in the narration
   - The visual is specified separately in the "visual" object
   - The narration should end naturally with the final teaching point
   - Do NOT include any meta-commentary or "that's all for today" style endings

OUTPUT FORMAT (respond with ONLY the JSON, no markdown code blocks):
{{
  "title": "Video Title Here",
  "metadata": {{
    "total_duration": "X:XX",
    "frame_count": N,
    "word_count": NNN,
    "target_wps": 2.5
  }},
  "frames": [
    {{
      "number": 0,
      "timing": {{
        "start": "0:00",
        "end": "0:20",
        "start_seconds": 0,
        "end_seconds": 20
      }},
      "word_count": 50,
      "narration": "Opening narration text here. This is ONLY the spoken text, no visual descriptions.",
      "visual": {{
        "type": "title",
        "reference": "Title slide with key concept preview"
      }}
    }},
    {{
      "number": 1,
      "timing": {{
        "start": "0:20",
        "end": "0:50",
        "start_seconds": 20,
        "end_seconds": 50
      }},
      "word_count": 75,
      "narration": "First teaching point narration here...",
      "visual": {{
        "type": "data_chart",
        "reference": "visual_1.png - VIX chart showing fear spikes"
      }}
    }},
    {{
      "number": 2,
      "timing": {{
        "start": "0:50",
        "end": "1:25",
        "start_seconds": 50,
        "end_seconds": 85
      }},
      "word_count": 88,
      "narration": "Second teaching point narration here...",
      "visual": {{
        "type": "conceptual",
        "reference": "Long vs Short position profit diagrams"
      }}
    }}
  ]
}}

Ensure word counts match timing (seconds × 2.5).
Reference each available data chart at least once.
Output ONLY valid JSON, no markdown formatting or code blocks.
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
- "You" only when action required

TTS PRONUNCIATION (CRITICAL for audio generation):
- NEVER use Unicode Greek letters (π, θ, α, β, etc.) in narration text
- Write Greek letters as English words for correct TTS pronunciation:
  * π → "pi" (the word, not the Greek letter)
  * θ → "theta"
  * α → "alpha"
  * β → "beta"
  * γ → "gamma"
  * δ → "delta"
  * σ → "sigma"
  * ω → "omega"
- In [Visual:] annotations, Unicode symbols are OK since they're not spoken
- Example: "sine of pi over 4" NOT "sine of π/4" """


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
) -> Dict:
    """Generate script using Claude, returns parsed JSON dict."""

    prompt = SCRIPT_GENERATION_PROMPT.format(
        video_brief=video_brief,
        chart_list=chart_list,
        conceptual_list=conceptual_list,
        style_guide=style_guide
    )

    response = client.generate(
        prompt=prompt,
        system="You are an expert educational script writer. Create clear, engaging narration that follows the teaching flow precisely. Match visuals to content. Be concise - no filler words. Output ONLY valid JSON.",
        max_tokens=8000,
        temperature=0.5
    )

    # Parse JSON response
    return parse_script_response(response)


def parse_script_response(response: str) -> Dict:
    """Parse Claude's JSON response into a dict."""
    # Clean up response - remove any markdown code blocks if present
    cleaned = response.strip()

    # Remove markdown code block markers if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {cleaned[:500]}...")


def json_to_script_data(json_data: Dict) -> ScriptData:
    """Convert parsed JSON dict to ScriptData object."""
    meta = json_data.get("metadata", {})
    total_duration_str = meta.get("total_duration", "0:00")

    metadata = ScriptMetadata(
        total_duration=total_duration_str,
        total_duration_seconds=parse_time_to_seconds(total_duration_str),
        frame_count=meta.get("frame_count", 0),
        word_count=meta.get("word_count", 0),
        target_wps=meta.get("target_wps", 2.5)
    )

    frames = []
    for frame_data in json_data.get("frames", []):
        timing = frame_data.get("timing", {})

        # Get timing values
        if "start_seconds" in timing:
            start_sec = timing["start_seconds"]
            end_sec = timing["end_seconds"]
        else:
            start_sec = parse_time_to_seconds(timing.get("start", "0:00"))
            end_sec = parse_time_to_seconds(timing.get("end", "0:00"))

        visual_data = frame_data.get("visual", {})
        visual = VisualInfo(
            type=visual_data.get("type", "conceptual"),
            reference=visual_data.get("reference", "")
        )

        frame = Frame(
            number=frame_data.get("number", 0),
            start_seconds=float(start_sec),
            end_seconds=float(end_sec),
            word_count=frame_data.get("word_count", 0),
            narration=frame_data.get("narration", ""),
            visual=visual
        )
        frames.append(frame)

    return ScriptData(
        title=json_data.get("title", "Untitled"),
        metadata=metadata,
        frames=frames
    )


def count_script_stats(script_data: ScriptData) -> Dict:
    """Count frames, words, and get duration from ScriptData."""
    return {
        "frames": len(script_data.frames),
        "words": script_data.metadata.word_count,
        "duration": script_data.metadata.total_duration
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
            script_json = generate_script(client, video_brief, chart_list, conceptual_list, style_guide)
            script_data = json_to_script_data(script_json)
        except Exception as e:
            print(f"  Error generating script: {e}")
            results["failed"].append(video_name)
            continue

        # Save both JSON (source of truth) and MD (for human review)
        save_script(script_data, video_dir, write_json=True, write_md=True)

        # Count stats
        stats = count_script_stats(script_data)

        print(f"  Saved: script.json (source of truth)")
        print(f"  Saved: script.md (for review)")
        print(f"    Frames: {stats['frames']}")
        print(f"    Words: {stats['words']}")
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
