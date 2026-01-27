#!/usr/bin/env python3
"""
Gemini API Client for Aurea Dicta
Wrapper for Google Gemini image generation (NanoBanana Pro)
"""

import os
import io
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

# Load environment variables (force reload to pick up changes)
load_dotenv(override=True)


class GeminiClient:
    """Client for Gemini 3 Pro Image API (Imagen 3 / NanoBanana Pro)"""

    def __init__(self, api_key: str = None):
        """
        Initialize Gemini client

        Args:
            api_key: Google Cloud API key. If None, reads from GOOGLE_CLOUD_API_KEY env var
        """
        self.api_key = api_key or os.getenv("GOOGLE_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_CLOUD_API_KEY not found. "
                "Set it in .env file or pass as parameter."
            )

        # Model configuration - Gemini 3 Pro Image (nanobanana pro)
        self.model = "gemini-3-pro-image-preview"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def generate_image(
        self,
        prompt: str,
        style: str = "hand-drawn educational",
        width: int = 1920,
        height: int = 1080
    ) -> bytes:
        """
        Generate an image using Gemini

        Args:
            prompt: Image generation prompt
            style: Visual style descriptor
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            PNG image bytes
        """
        # Build styled prompt
        styled_prompt = f"""Generate a {style} style image:

{prompt}

Style requirements:
- Clean, sketch-like illustration
- Educational whiteboard aesthetic
- Simple and clear visuals
- Warm white background (#FAFAF9)
- Use blue (#3B82F6) for key concepts
- Use orange (#F97316) for highlights
- No photorealistic elements
- Minimal text (if any)
- High contrast, readable at 1080p
"""

        url = f"{self.base_url}/models/{self.model}:generateContent"
        params = {"key": self.api_key}

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": styled_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
                "imageConfig": {
                    "aspectRatio": "16:9"
                }
            }
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(url, params=params, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Gemini API error: {response.status_code} - {response.text}")

        result = response.json()

        # Extract image from response
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            parts = candidate.get("content", {}).get("parts", [])

            # Check for finish reason that might explain no image
            finish_reason = candidate.get("finishReason", "")

            for part in parts:
                if "inlineData" in part:
                    image_data = part["inlineData"]["data"]
                    raw_bytes = base64.b64decode(image_data)

                    # Convert to PNG using PIL (API may return JPEG)
                    img = Image.open(io.BytesIO(raw_bytes))
                    png_buffer = io.BytesIO()
                    img.save(png_buffer, format="PNG")
                    return png_buffer.getvalue()

            # No image found - provide diagnostic info
            text_parts = [p.get("text", "") for p in parts if "text" in p]
            text_response = " ".join(text_parts)[:500] if text_parts else "(no text)"

            # Check for safety ratings
            safety_ratings = candidate.get("safetyRatings", [])
            blocked_categories = [
                r.get("category", "") for r in safety_ratings
                if r.get("probability", "") in ("HIGH", "MEDIUM") or r.get("blocked", False)
            ]

            error_msg = f"No image in response. finishReason={finish_reason}"
            if blocked_categories:
                error_msg += f", blocked_categories={blocked_categories}"
            if text_response and text_response != "(no text)":
                error_msg += f", text_response={text_response[:200]}"

            raise Exception(error_msg)

        # No candidates at all
        # Check for prompt feedback (content filtered before generation)
        prompt_feedback = result.get("promptFeedback", {})
        block_reason = prompt_feedback.get("blockReason", "")
        safety_ratings = prompt_feedback.get("safetyRatings", [])

        if block_reason:
            raise Exception(f"Prompt blocked: {block_reason}, safety={safety_ratings}")

        raise Exception(f"No candidates in response: {result}")

    def generate_slide(
        self,
        frame_type: str,
        content: dict,
        frame_number: int
    ) -> bytes:
        """
        Generate a slide image for a specific frame

        Args:
            frame_type: Type of slide (title_slide, concept_slide, etc.)
            content: Dictionary with slide content
            frame_number: Frame number for logging

        Returns:
            PNG image bytes
        """
        prompts = {
            "title_slide": self._build_title_prompt,
            "quote_slide": self._build_quote_prompt,
            "concept_slide": self._build_concept_prompt,
            "comparison_slide": self._build_comparison_prompt,
            "example_slide": self._build_example_prompt,
            "summary_slide": self._build_summary_prompt,
        }

        builder = prompts.get(frame_type, self._build_concept_prompt)
        prompt = builder(content)

        print(f"Generating frame {frame_number} ({frame_type})...")
        return self.generate_image(prompt)

    def _build_title_prompt(self, content: dict) -> str:
        title = content.get("title", "Untitled")
        subtitle = content.get("subtitle", "")
        return f"""Title slide for educational video:

Title: "{title}"
{f'Subtitle: "{subtitle}"' if subtitle else ''}

Design:
- Large, bold title text centered
- Clean, minimal design
- Educational feel
- Visual metaphor related to the topic (subtle, supporting)
- NO clutter
"""

    def _build_quote_prompt(self, content: dict) -> str:
        quote = content.get("quote", "")
        attribution = content.get("attribution", "")
        return f"""Quote slide for educational video:

Quote: "{quote}"
Attribution: "— {attribution}"

Design:
- Quote text prominently displayed (italic if possible)
- Attribution smaller, right-aligned
- Simple decorative element (quotation marks or subtle icon)
- Clean, inspirational feel
- Plenty of whitespace
"""

    def _build_concept_prompt(self, content: dict) -> str:
        concept = content.get("concept", "")
        elements = content.get("elements", [])
        return f"""Concept slide for educational video:

Main concept: {concept}
Key elements: {', '.join(elements)}

Design:
- Clear visual representation of the concept
- Hand-drawn illustration style
- Maximum 3 visual elements
- Supporting icons or diagrams
- Educational, not corporate
"""

    def _build_comparison_prompt(self, content: dict) -> str:
        left = content.get("left", {})
        right = content.get("right", {})
        return f"""Comparison slide for educational video:

Left side: {left.get('label', 'A')} - {left.get('description', '')}
Right side: {right.get('label', 'B')} - {right.get('description', '')}

Design:
- Two-column layout
- Clear visual distinction between sides
- Use color to differentiate (blue vs orange)
- Simple divider in center
- Equal visual weight on each side
"""

    def _build_example_prompt(self, content: dict) -> str:
        scenario = content.get("scenario", "")
        data = content.get("data", "")
        return f"""Example slide for educational video:

Scenario: {scenario}
Data/Numbers: {data}

Design:
- Real-world scenario illustration
- Numbers prominently displayed
- Visual representation of the outcome
- Hand-drawn style with clear labels
- Educational and practical feel
"""

    def _build_summary_prompt(self, content: dict) -> str:
        takeaways = content.get("takeaways", [])
        return f"""Summary slide for educational video:

Key takeaways:
{chr(10).join([f'- {t}' for t in takeaways])}

Design:
- 3 key points with checkmarks or icons
- Clean, organized layout
- Sense of completion
- Optional: subtle visual metaphor
- Call to action feel
"""

    def save_image(self, image_bytes: bytes, filepath: str) -> str:
        """
        Save image bytes to file as PNG

        Args:
            image_bytes: Image data (any format PIL can read)
            filepath: Output file path

        Returns:
            Saved file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Use PIL to ensure proper PNG format
        img = Image.open(io.BytesIO(image_bytes))
        img.save(path, format="PNG")

        return str(path)


def main():
    """Test Gemini client"""
    client = GeminiClient()

    # Test image generation
    try:
        image = client.generate_image(
            "A simple illustration of a lightbulb representing ideas and learning",
            style="hand-drawn educational"
        )
        client.save_image(image, "test_output.png")
        print("Test image saved to test_output.png")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
