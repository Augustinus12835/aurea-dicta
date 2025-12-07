#!/usr/bin/env python3
"""
Transcript Cleaning - Extract educational content only.
No timestamps, no structure - just clean educational prose.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils.claude_client import ClaudeClient


CLEANING_PROMPT = """Extract the educational content from this lecture transcript.

REMOVE completely:
- Course administration (syllabus, grading, assignments, deadlines, office hours)
- Housekeeping ("can you hear me", "let me share my screen", "we'll take a break")
- Filler words and verbal tics (um, uh, like, you know, basically, essentially, right?)
- Redundant explanations (keep the clearest version only)
- Student questions about logistics
- Off-topic personal anecdotes

KEEP and express clearly:
- Concept definitions and explanations
- Key terminology
- Examples that illustrate concepts
- Calculations and problem-solving steps
- Important insights and takeaways
- Real-world applications

OUTPUT:
Clean, flowing prose organized by topic.
Write as if explaining to an engaged student.
Preserve all technical accuracy.

---

LECTURE TRANSCRIPT:
{transcript}

---

EDUCATIONAL CONTENT:"""


def extract_full_text(transcript: dict) -> str:
    """Extract all text from transcript - either from segments or full text."""
    # Try segments first
    segments = transcript.get("segments", [])
    if segments:
        return " ".join(seg.get("text", "").strip() for seg in segments)
    # Fall back to full text
    return transcript.get("text", "")


def chunk_text(text: str, max_chars: int = 25000) -> list:
    """Split text into chunks for processing."""
    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(word)
        current_len += len(word) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def clean_transcript(client: ClaudeClient, text: str, verbose: bool = False) -> str:
    """Clean transcript, handling chunking if needed."""

    # If small enough, process in one go
    if len(text) < 30000:
        if verbose:
            print(f"  Processing {len(text):,} chars in single pass...")

        response = client.generate(
            prompt=CLEANING_PROMPT.format(transcript=text),
            system="Extract educational content. Be thorough but concise.",
            max_tokens=8000,
            temperature=0.3
        )
        return response

    # Otherwise, chunk and process
    chunks = chunk_text(text, max_chars=25000)
    if verbose:
        print(f"  Processing {len(chunks)} chunks...")

    cleaned_parts = []
    for i, chunk in enumerate(chunks):
        if verbose:
            print(f"    Chunk {i+1}/{len(chunks)}: {len(chunk):,} chars")

        response = client.generate(
            prompt=CLEANING_PROMPT.format(transcript=chunk),
            system="Extract educational content. Be thorough but concise.",
            max_tokens=6000,
            temperature=0.3
        )
        cleaned_parts.append(response)

    return "\n\n".join(cleaned_parts)


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_transcript.py <pipeline_dir> [--verbose]")
        print()
        print("Extracts educational content from transcript, removing:")
        print("  - Course administration and logistics")
        print("  - Filler words and verbal tics")
        print("  - Redundant explanations")
        print("  - Off-topic tangents")
        print()
        print("Example:")
        print("  python clean_transcript.py pipeline/YOUR_LECTURE --verbose")
        sys.exit(1)

    pipeline_dir = sys.argv[1]
    verbose = "--verbose" in sys.argv

    print("=" * 60)
    print("Transcript Cleaning")
    print("=" * 60)

    # Load transcript
    transcript_path = Path(pipeline_dir) / "transcript.json"
    if not transcript_path.exists():
        print(f"Error: {transcript_path} not found")
        sys.exit(1)

    with open(transcript_path) as f:
        transcript = json.load(f)

    # Extract text
    full_text = extract_full_text(transcript)
    print(f"Original: {len(full_text):,} characters")

    # Clean
    print("\nCleaning transcript...")
    client = ClaudeClient()
    cleaned = clean_transcript(client, full_text, verbose)

    print(f"\nCleaned:  {len(cleaned):,} characters")
    reduction = (1 - len(cleaned) / len(full_text)) * 100 if full_text else 0
    print(f"Reduction: {reduction:.0f}%")

    # Save
    output_path = Path(pipeline_dir) / "content_cleaned.txt"
    with open(output_path, "w") as f:
        f.write(cleaned)

    print(f"\nSaved: {output_path}")
    print(f"\nNext: python segment_concepts.py {pipeline_dir}")


if __name__ == "__main__":
    main()
