#!/usr/bin/env python3
"""
Claude API Client for Aurea Dicta
Wrapper for Anthropic Claude Opus 4.5 API calls
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    raise


class ClaudeClient:
    """Client for Claude Opus 4.5 API interactions"""

    def __init__(self, api_key: str = None):
        """
        Initialize Claude client

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. "
                "Set it in .env file or pass as parameter."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-opus-4-5-20251101"

    def call_claude(
        self,
        prompt: str,
        system: str,
        max_tokens: int = 8000,
        temperature: float = 0.3
    ) -> str:
        """
        Generic Claude API call for any prompt.

        This is a simplified interface for classification/grouping prompts.

        Args:
            prompt: User prompt (the full prompt with data)
            system: System prompt (role/instruction)
            max_tokens: Maximum output tokens
            temperature: Lower = more deterministic (0.3 for structured output)

        Returns:
            Generated text content
        """
        return self.generate(prompt, system=system, max_tokens=max_tokens, temperature=temperature)

    def generate(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 8192,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using Claude

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_tokens: Maximum output tokens
            temperature: Creativity parameter (0-1)

        Returns:
            Generated text content
        """
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        if temperature != 1.0:
            kwargs["temperature"] = temperature

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def segment_lecture(self, transcript: str, target_videos: int = 10) -> str:
        """
        Segment a lecture transcript into concept videos

        Args:
            transcript: Full lecture transcript
            target_videos: Target number of videos (8-10 typically)

        Returns:
            JSON string with segmentation data
        """
        system = """You are an expert educational content designer.
Your task is to analyze lecture transcripts and segment them into logical concept videos.
Each video should be 5-8 minutes of content (750-1200 words at 2.5 words/second).
Always respond with valid JSON."""

        prompt = f"""Analyze this lecture transcript and segment it into {target_videos} concept videos.

REQUIREMENTS:
1. Each video: 5-8 minutes of content
2. Group related concepts together
3. Create logical progression (foundational → advanced)
4. Identify natural breakpoints
5. Extract key takeaways for each segment

TRANSCRIPT:
{transcript[:50000]}  # Truncate if very long

OUTPUT FORMAT (JSON):
{{
    "total_videos": N,
    "segments": [
        {{
            "video_number": 1,
            "title": "Clear, descriptive title",
            "duration_target": "X minutes",
            "start_timestamp": "HH:MM:SS",
            "end_timestamp": "HH:MM:SS",
            "key_concepts": ["concept1", "concept2", "concept3"],
            "key_takeaways": ["takeaway1", "takeaway2", "takeaway3"],
            "content_scope": "Brief description of what this video covers"
        }}
    ]
}}

Respond with ONLY valid JSON, no other text."""

        return self.generate(prompt, system=system, temperature=0.3)

    def generate_video_brief(
        self,
        segment_info: dict,
        transcript_section: str,
        template: str = None
    ) -> str:
        """
        Generate a video brief from segment information

        Args:
            segment_info: Dictionary with segment metadata
            transcript_section: Relevant portion of transcript
            template: Optional video brief template

        Returns:
            Markdown video brief content
        """
        system = """You are an educational content strategist.
Create concise, actionable video briefs that guide production.
Focus on teaching flow, key examples, and visual requirements."""

        prompt = f"""Create a video brief for this educational video:

SEGMENT INFO:
Title: {segment_info.get('title', 'Untitled')}
Duration: {segment_info.get('duration_target', '6 minutes')}
Key Concepts: {', '.join(segment_info.get('key_concepts', []))}
Takeaways: {', '.join(segment_info.get('key_takeaways', []))}

TRANSCRIPT SECTION:
{transcript_section[:10000]}

Create a video brief in markdown format including:
1. Core Question (what problem does this solve?)
2. Key Concepts (2-3 bullet points)
3. Teaching Flow (Hook → Build → Deepen → Apply)
4. Must-Include Examples (3 real-world examples)
5. Visual Requirements (diagrams to generate)
6. Common Misconceptions (2-3 mistakes learners make)

Be concise and actionable."""

        return self.generate(prompt, system=system, temperature=0.5)

    def generate_script(
        self,
        video_brief: str,
        transcript_section: str,
        duration_minutes: int,
        style_guide: str = None
    ) -> str:
        """
        Generate a narration script from a video brief

        Args:
            video_brief: The video brief content
            transcript_section: Relevant transcript section
            duration_minutes: Target video duration
            style_guide: Optional teaching style guide

        Returns:
            Markdown script with frame timing
        """
        word_count = int(duration_minutes * 60 * 2.5)

        system = """You are an expert educational scriptwriter.
Write concise, engaging narration scripts for educational videos.
Follow the exact frame format specified. No filler words. Active voice only."""

        prompt = f"""Write a narration script for this educational video.

VIDEO BRIEF:
{video_brief}

ORIGINAL LECTURE EXCERPT:
{transcript_section[:8000]}

{f"STYLE GUIDE: {style_guide}" if style_guide else ""}

REQUIREMENTS:
- Duration: {duration_minutes} minutes ({word_count} words max at 2.5 words/second)
- Format: Use frame headers exactly as shown below
- Style: Conversational, no filler words, active voice
- Each frame: 10-60 seconds, clear single concept

OUTPUT FORMAT:
## Frame 0 (0:00-0:15) • 38 words

[Narration text for frame 0]

---

## Frame 1 (0:15-0:30) • 38 words

[Narration text for frame 1]

---

Continue for all frames. Final frame should include closing summary.
End with a Summary Statistics section showing total words and timing."""

        return self.generate(prompt, system=system, temperature=0.6)

    def analyze_script_for_visuals(self, script: str) -> str:
        """
        Analyze a script and determine visual requirements for each frame

        Args:
            script: The narration script

        Returns:
            JSON with visual requirements per frame
        """
        system = """You are a visual designer for educational content.
Analyze scripts and specify what visuals are needed for each frame.
Consider hand-drawn educational aesthetics."""

        prompt = f"""Analyze this script and specify visual requirements for each frame.

SCRIPT:
{script}

For each frame, specify:
- frame_type: title_slide, quote_slide, concept_slide, comparison_slide, example_slide, or summary_slide
- visual_elements: List of what should appear
- diagram_needed: true/false
- diagram_description: If needed, describe the diagram
- data_chart: If data visualization needed, specify type and data source

OUTPUT FORMAT (JSON):
{{
    "frames": [
        {{
            "frame_number": 0,
            "frame_type": "title_slide",
            "visual_elements": ["title text", "subtitle"],
            "diagram_needed": false,
            "diagram_description": null,
            "data_chart": null
        }},
        ...
    ]
}}

Respond with ONLY valid JSON."""

        return self.generate(prompt, system=system, temperature=0.3)


    def verify_math(
        self,
        narration: str,
        visual_context: str,
        frame_number: int = 0,
        budget_tokens: int = 10000
    ) -> dict:
        """
        Verify mathematical content using extended thinking.

        Uses Claude's extended thinking capability to carefully work through
        mathematical calculations, then returns:
        - natural_narration: TTS-friendly version of the content
        - math_steps: Precise mathematical notation for Gemini slides
        - verification_status: "correct", "corrected", or "unclear"

        Args:
            narration: The original narration text with math content
            visual_context: Description of what the visual should show
            frame_number: Frame number for context
            budget_tokens: Extended thinking budget (default 10000)

        Returns:
            dict with natural_narration, math_steps, verification_status, etc.
        """
        system = """You are a mathematics verification expert. Your task is to:
1. Verify that any mathematical calculations in the narration are correct
2. Generate a natural English narration suitable for text-to-speech (no symbols like √, ∫, etc.)
3. Generate precise mathematical steps for educational slides

You must respond with ONLY valid JSON, no other text."""

        prompt = f"""Analyze this educational math content and verify its accuracy.

FRAME NUMBER: {frame_number}

ORIGINAL NARRATION:
{narration}

VISUAL CONTEXT:
{visual_context}

TASK:
1. Work through any mathematical calculations step-by-step to verify correctness
2. Create a natural English narration that:
   - Reads well for text-to-speech (spell out symbols: "square root of x" not "√x")
   - Uses natural phrases like "two plus two" instead of "2 + 2"
   - Maintains the teaching flow and meaning
   - Can be longer than the original if needed for clear step-by-step explanation (this is math-intensive content)

3. Create precise math steps that:
   - Show the complete mathematical process
   - Use proper LaTeX-style notation
   - Include intermediate steps
   - Highlight the final answer

RESPOND WITH THIS EXACT JSON FORMAT:
{{
    "verification_status": "correct" or "corrected" or "unclear",
    "issues_found": ["list of any errors found, empty if none"],
    "natural_narration": "The TTS-friendly narration text...",
    "math_steps": [
        {{
            "step": 1,
            "expression": "LaTeX expression",
            "operation": "What operation is being done",
            "note": "Optional note about this step"
        }}
    ],
    "final_answer": "The final numerical or symbolic answer",
    "confidence": "high" or "medium" or "low"
}}

Important:
- If no math is present, set verification_status to "unclear" and return the original narration
- The natural_narration can be longer than the original for thorough explanation
- Math steps should be thorough enough for a student to follow along"""

        try:
            # Use extended thinking for thorough verification
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                },
                messages=[{"role": "user", "content": prompt}],
                system=system
            )

            # Extract the text response (extended thinking puts result in content blocks)
            result_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    result_text = block.text
                    break

            # Parse JSON from response
            # Try to find JSON in the response
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON from the text
                import re
                json_match = re.search(r'\{[\s\S]*\}', result_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from response: {result_text[:500]}")

            return result

        except Exception as e:
            # Return a safe fallback that preserves the original
            return {
                "verification_status": "error",
                "issues_found": [str(e)],
                "natural_narration": narration,
                "math_steps": [],
                "final_answer": None,
                "confidence": "low",
                "error": str(e)
            }


def main():
    """Test Claude client"""
    client = ClaudeClient()

    # Simple test
    response = client.generate("Say 'Claude client working!' in exactly 3 words.")
    print(f"Test response: {response}")


if __name__ == "__main__":
    main()
