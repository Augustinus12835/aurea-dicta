#!/usr/bin/env python3
"""
Video Brief Generation for Aurea Dicta

Generates video_brief.md with:
- Pedagogical structure (Hook → Build → Deepen → Apply)
- Specific visual requirements with data specifications
- Machine-parseable visual_specs.json for diagram generation

Uses cleaned content from Video-N/content.txt (no timestamps needed).
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.claude_client import ClaudeClient


BRIEF_GENERATION_PROMPT = """You are creating a video brief for an educational video.

VIDEO INFORMATION:
Title: {title}
Core Concept: {core_concept}
Duration Estimate: {duration_estimate}

KEY TAKEAWAYS:
{key_takeaways}

EXAMPLES TO INCLUDE:
{examples}

SOURCE CONTENT:
{content}

---

Create a comprehensive video brief with TWO outputs:

## OUTPUT 1: VIDEO BRIEF (Markdown)

Create a production-ready brief following this structure:

### Video Metadata
- Title, duration, core question (what problem does this solve?)

### Key Concepts (2-3 bullets)
- The essential ideas viewers must understand

### Teaching Flow
1. **Hook:** (15-30 seconds) What counterintuitive insight, surprising fact, or relatable scenario opens the video? Be specific.
2. **Build:** (2-3 minutes) What foundational concepts must be established? In what order?
3. **Deepen:** (2-3 minutes) What nuance, edge cases, or common confusions should be addressed?
4. **Apply:** (1-2 minutes) What practical example demonstrates mastery? What can viewers do with this knowledge?

### Must-Include Examples
- List 2-3 specific examples with brief descriptions
- Note which teaching flow section each belongs to

### Visual Requirements
For EACH visual needed, specify:

**[Visual Name]**
- Type: [data_chart | conceptual_diagram | comparison_table | formula_walkthrough]
- Purpose: What does this visual teach?
- Teaching flow: Which section uses this?
- If data_chart:
  - Data source: yfinance or FRED
  - Tickers/Series: Specific identifiers (e.g., "^GSPC", "VIXCLS", "DGS10")
  - Time range: e.g., "2020-01-01 to 2024-12-31" or "5Y"
  - Chart type: line, bar, scatter, comparison
  - Key annotations: What to highlight (e.g., "mark COVID crash", "show 2008 peak")
- If conceptual_diagram:
  - Elements: What shapes/components (e.g., "2x2 matrix with quadrants labeled...")
  - Style: hand-drawn sketch, clean diagram, flowchart
- If formula_walkthrough:
  - Formula: The actual equation
  - Variables: What each represents
  - Example calculation: With real numbers

### Common Misconceptions
- 2-3 things learners typically get wrong
- How to address each in the video

### Script Notes
- Tone guidance
- Technical terms to define
- Transitions between sections

---

## OUTPUT 2: VISUAL SPECIFICATIONS (JSON)

After the markdown brief, output a JSON block with machine-parseable visual specs:

```json
{{
  "requires_math": false,
  "visuals": [
    {{
      "id": "visual_1",
      "name": "S&P 500 Volatility During Crises",
      "type": "data_chart",
      "purpose": "Show how volatility spikes during market stress",
      "teaching_section": "build",
      "data_source": "yfinance",
      "tickers": ["^VIX"],
      "time_range": {{"start": "2018-01-01", "end": "2024-12-31"}},
      "chart_type": "line",
      "annotations": [
        {{"date": "2020-03-16", "label": "COVID crash", "style": "vertical_line"}},
        {{"date": "2022-01-01", "label": "Rate hike fears", "style": "vertical_line"}}
      ],
      "styling": {{
        "title": "VIX Index: Fear Gauge of the Market",
        "y_label": "VIX Level",
        "highlight_above": 30,
        "color_scheme": "fear"
      }}
    }},
    {{
      "id": "visual_2",
      "name": "Risk vs Return Tradeoff",
      "type": "conceptual_diagram",
      "purpose": "Illustrate that higher expected returns require accepting higher risk",
      "teaching_section": "deepen",
      "elements": [
        "x-axis: Risk (Standard Deviation)",
        "y-axis: Expected Return",
        "scatter points for: T-bills, Bonds, Large-cap stocks, Small-cap stocks, Emerging markets",
        "upward sloping trend line"
      ],
      "style": "clean_academic"
    }}
  ]
}}
```

**requires_math field (REQUIRED):**
- Set to `true` if this video involves:
  - Mathematical calculations, formulas, or derivations
  - Quantitative analysis (statistics, financial calculations)
  - Graphs with numerical data and equations
  - Step-by-step mathematical proofs or solutions
- Set to `false` if this video is:
  - Conceptual/theoretical (history, religion, philosophy, literature)
  - Narrative/story-based content
  - Qualitative analysis without numerical formulas
  - Maps, timelines, comparison tables without equations

Example: A finance video about "How the GFC happened" (conceptual) → requires_math: false
Example: A finance video about "Present Value calculations" → requires_math: true

IMPORTANT for visual specs:
- Use REAL ticker symbols (^GSPC, ^VIX, AAPL) and FRED series IDs (DGS10, VIXCLS, CPIAUCSL)
- Be specific about time ranges
- Annotations should reference actual historical events
- Every visual must have a clear teaching purpose
"""


def load_segments(pipeline_dir: str) -> dict:
    """Load segments.json from pipeline directory."""
    segments_path = Path(pipeline_dir) / "segments.json"

    if not segments_path.exists():
        raise FileNotFoundError(f"segments.json not found. Run segment_concepts.py first.")

    with open(segments_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_video_content(video_dir: Path) -> str:
    """Load content.txt from video directory."""
    content_path = video_dir / "content.txt"

    if content_path.exists():
        with open(content_path, "r", encoding="utf-8") as f:
            return f.read()

    return ""


def parse_brief_and_specs(response: str) -> tuple:
    """
    Parse LLM response into brief markdown and visual specs JSON.

    Returns:
        (brief_markdown, visual_specs_dict)
    """
    brief_md = response
    visual_specs = None

    # Try to extract JSON block
    if "```json" in response:
        parts = response.split("```json")
        brief_md = parts[0].strip()

        if len(parts) > 1:
            json_part = parts[1].split("```")[0].strip()
            try:
                visual_specs = json.loads(json_part)
            except json.JSONDecodeError:
                print("    Warning: Could not parse visual specs JSON")

    return brief_md, visual_specs


def generate_brief(client: ClaudeClient, video: dict, content: str) -> tuple:
    """
    Generate video brief with visual specifications.

    Returns:
        (brief_markdown, visual_specs_dict)
    """
    # Format key takeaways
    takeaways = video.get("key_takeaways", [])
    if isinstance(takeaways, list):
        takeaways_str = "\n".join(f"- {t}" for t in takeaways)
    else:
        takeaways_str = str(takeaways)

    # Format examples
    examples = video.get("examples", [])
    if isinstance(examples, list):
        examples_str = "\n".join(f"- {e}" for e in examples)
    else:
        examples_str = str(examples) if examples else "None specified"

    prompt = BRIEF_GENERATION_PROMPT.format(
        title=video.get("title", "Untitled"),
        core_concept=video.get("core_concept", ""),
        duration_estimate=video.get("duration_estimate", "6-8 minutes"),
        key_takeaways=takeaways_str,
        examples=examples_str,
        content=content[:12000]  # Limit content size
    )

    response = client.generate(
        prompt=prompt,
        system="You are an expert educational content designer. Create detailed, actionable video briefs with specific visual requirements.",
        max_tokens=8000,
        temperature=0.5
    )

    return parse_brief_and_specs(response)


def main():
    """Main brief generation function."""
    if len(sys.argv) < 2:
        print("Usage: python generate_briefs.py <pipeline_dir> [--video N]")
        print()
        print("Generates video briefs from cleaned content (no timestamps needed).")
        print("Each brief includes pedagogical structure and visual specifications.")
        print()
        print("Requires: segments.json and Video-N/content.txt (from segment_concepts.py)")
        print()
        print("Options:")
        print("  --video N     Generate brief for specific video only")
        print("  --verbose     Show detailed output")
        print()
        print("Examples:")
        print("  python generate_briefs.py pipeline/YOUR_LECTURE")
        print("  python generate_briefs.py pipeline/lecture --video 3")
        sys.exit(1)

    pipeline_dir = sys.argv[1]
    specific_video = None
    verbose = "--verbose" in sys.argv

    if "--video" in sys.argv:
        idx = sys.argv.index("--video")
        if idx + 1 < len(sys.argv):
            specific_video = int(sys.argv[idx + 1])

    print("=" * 60)
    print("Video Brief Generation")
    print("=" * 60)
    print(f"Pipeline: {pipeline_dir}")

    try:
        # Load segments
        segments = load_segments(pipeline_dir)

        # Initialize Claude
        client = ClaudeClient()

        # Get videos to process
        videos = segments.get("videos", [])
        if specific_video:
            videos = [v for v in videos if v.get("number") == specific_video]

        if not videos:
            print("No videos to process")
            sys.exit(1)

        print(f"\nProcessing {len(videos)} video(s)...")

        all_visual_specs = {}

        for video in videos:
            video_num = video.get("number", 0)
            title = video.get("title", "Untitled")

            print(f"\nVideo {video_num}: {title}")

            # Setup video directory
            video_dir = Path(pipeline_dir) / f"Video-{video_num}"
            video_dir.mkdir(parents=True, exist_ok=True)

            # Load content - prefer content.txt, fallback to segment content
            content = load_video_content(video_dir)
            if not content:
                content = video.get("content", "")
                # Save content.txt if it doesn't exist
                if content:
                    with open(video_dir / "content.txt", "w") as f:
                        f.write(content)

            if not content:
                print(f"  Warning: No content found for Video {video_num}")
                continue

            print(f"  Content: {len(content):,} chars")

            # Generate brief
            print(f"  Generating brief...")
            brief_md, visual_specs = generate_brief(client, video, content)

            # Save brief
            brief_path = video_dir / "video_brief.md"
            with open(brief_path, "w", encoding="utf-8") as f:
                f.write(f"# Video {video_num}: {title}\n\n")
                f.write(brief_md)
            print(f"  Saved: {brief_path}")

            # Save visual specs
            if visual_specs:
                specs_path = video_dir / "visual_specs.json"
                with open(specs_path, "w", encoding="utf-8") as f:
                    json.dump(visual_specs, f, indent=2)
                print(f"  Saved: {specs_path}")
                print(f"  Visuals: {len(visual_specs.get('visuals', []))} specified")

                all_visual_specs[f"Video-{video_num}"] = visual_specs
            else:
                print(f"  Warning: No visual specs generated")

            # Create diagrams directory
            diagrams_dir = video_dir / "diagrams"
            diagrams_dir.mkdir(exist_ok=True)

        # Summary
        print("\n" + "=" * 60)
        print("Brief Generation Complete")
        print("=" * 60)

        total_visuals = sum(
            len(specs.get("visuals", []))
            for specs in all_visual_specs.values()
        )
        print(f"\nTotal visuals to generate: {total_visuals}")
        print(f"\nNext step: python generate_diagrams.py {pipeline_dir}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
