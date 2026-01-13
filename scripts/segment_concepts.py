#!/usr/bin/env python3
"""
Concept Segmentation - Divide cleaned content into videos.
Each video gets its content directly - no references needed.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils.claude_client import ClaudeClient


SEGMENTATION_PROMPT = """Divide this educational content into concept videos.

Each video should:
- Focus on ONE core concept or closely related set of concepts
- Contain enough material for a 5-8 minute produced video
- Stand alone (viewer shouldn't need to watch other videos first)

CONTENT:
{content}

---

Create the video segments. For each video:

1. **title**: Clear, searchable YouTube-style title
2. **core_concept**: One sentence - what will viewer understand after watching?
3. **content**: Copy the relevant paragraphs for this video (this becomes source material)
4. **key_takeaways**: 2-3 bullet points
5. **examples**: List any examples/calculations that should be included
6. **duration_estimate**: Estimated video length based on content density

Rules:
- Quality over quantity (4 good videos > 8 thin videos)
- Don't stretch content to fill videos
- Don't duplicate content across videos
- Group naturally related concepts

OUTPUT as valid JSON:
{{
  "video_count": N,
  "videos": [
    {{
      "number": 1,
      "title": "...",
      "core_concept": "...",
      "content": "... actual paragraphs of content ...",
      "key_takeaways": ["...", "..."],
      "examples": ["...", "..."],
      "duration_estimate": "X minutes"
    }}
  ],
  "unused_content": "... any content that didn't fit into videos ...",
  "notes": "... observations about segmentation decisions ..."
}}"""


def main():
    if len(sys.argv) < 2:
        print("Usage: python segment_concepts.py <pipeline_dir>")
        print()
        print("Divides cleaned content into concept videos.")
        print("Each video gets its actual content directly.")
        print()
        print("Requires: content_cleaned.txt (from clean_transcript.py)")
        print()
        print("Example:")
        print("  python segment_concepts.py pipeline/YOUR_LECTURE")
        sys.exit(1)

    pipeline_dir = sys.argv[1]

    print("=" * 60)
    print("Concept Segmentation")
    print("=" * 60)

    # Load cleaned content
    content_path = Path(pipeline_dir) / "content_cleaned.txt"
    if not content_path.exists():
        print(f"Error: {content_path} not found")
        print("Run clean_transcript.py first")
        sys.exit(1)

    with open(content_path) as f:
        content = f.read()

    print(f"Content: {len(content):,} characters")

    # Segment
    print("\nSegmenting into concept videos...")
    client = ClaudeClient()

    response = client.generate(
        prompt=SEGMENTATION_PROMPT.format(content=content),
        system="You are segmenting educational content into concept videos. Output valid JSON only.",
        max_tokens=16000,
        temperature=0.4
    )

    # Parse JSON
    try:
        # Handle markdown code blocks if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        segments = json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}")
        print("Raw response saved to segments_raw.txt")
        with open(Path(pipeline_dir) / "segments_raw.txt", "w") as f:
            f.write(response)
        sys.exit(1)

    # Save segments
    segments_path = Path(pipeline_dir) / "segments.json"
    with open(segments_path, "w") as f:
        json.dump(segments, f, indent=2)

    video_count = segments.get('video_count', len(segments.get('videos', [])))
    print(f"\nCreated {video_count} videos")

    # Create video folders with content
    for video in segments.get("videos", []):
        num = video.get("number", 0)
        video_dir = Path(pipeline_dir) / f"Video-{num}"
        video_dir.mkdir(exist_ok=True)

        # Save this video's content
        content_file = video_dir / "content.txt"
        with open(content_file, "w") as f:
            f.write(f"# {video.get('title', 'Untitled')}\n\n")
            f.write(f"**Core Concept:** {video.get('core_concept', '')}\n\n")
            f.write("---\n\n")
            f.write(video.get("content", ""))
            f.write("\n\n---\n\n")
            f.write("**Key Takeaways:**\n")
            for takeaway in video.get("key_takeaways", []):
                f.write(f"- {takeaway}\n")

        print(f"  Video {num}: {video.get('title', 'Untitled')}")
        print(f"           Duration: {video.get('duration_estimate', 'N/A')}")

    print(f"\nSaved: {segments_path}")
    print(f"\nNext: python generate_briefs.py {pipeline_dir}")


if __name__ == "__main__":
    main()
