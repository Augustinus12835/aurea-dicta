#!/usr/bin/env python3
"""
Study Guide Generation for Aurea Dicta

Generates PDF study guides from video frames and scripts:
1. slides.pdf - All frames as a slideshow (frames only)
2. study_guide.pdf - Frames + transcript text for reading

Usage:
    python generate_study_guide.py <video_dir>
    python generate_study_guide.py pipeline/BANK5016_Week1/Video-1

Output:
    Video-N/slides.pdf
    Video-N/study_guide.pdf
"""

import sys
import re
import io
import tempfile
from pathlib import Path

try:
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    from PIL import Image
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: pip install reportlab pillow")
    sys.exit(1)


# Page dimensions (landscape A4)
PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)  # 297mm × 210mm

# Layout proportions for study guide (frames + text)
HEADER_HEIGHT = PAGE_HEIGHT * 0.06
IMAGE_HEIGHT = PAGE_HEIGHT * 0.55
TEXT_HEIGHT = PAGE_HEIGHT * 0.34
FOOTER_HEIGHT = PAGE_HEIGHT * 0.05
MARGIN = 15 * mm

# Image compression settings
IMAGE_MAX_WIDTH = 1280   # Resize to 720p equivalent
JPEG_QUALITY = 80        # Good quality, small size

# Colors
HEADER_COLOR = HexColor("#1F2937")
TEXT_COLOR = HexColor("#000000")  # Black text
FOOTER_COLOR = HexColor("#6B7280")


def compress_image(image_path: Path, temp_dir: Path) -> Path:
    """
    Compress an image for PDF embedding.
    Returns path to compressed temp file.
    """
    with Image.open(image_path) as img:
        # Convert to RGB if necessary (for JPEG)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        # Resize if too large
        if img.width > IMAGE_MAX_WIDTH:
            ratio = IMAGE_MAX_WIDTH / img.width
            new_size = (IMAGE_MAX_WIDTH, int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save as compressed JPEG
        compressed_path = temp_dir / f"{image_path.stem}_compressed.jpg"
        img.save(compressed_path, 'JPEG', quality=JPEG_QUALITY, optimize=True)

        return compressed_path


def parse_script(script_path: Path) -> tuple:
    """
    Parse script.md into list of frames with timing and text.

    Returns:
        (video_title, list of frame dicts)
    """
    content = script_path.read_text()
    frames = []

    # Extract video title
    title_match = re.search(r'^# Script: (.+)$', content, re.MULTILINE)
    video_title = title_match.group(1) if title_match else "Video"

    # Match frame headers and content
    # Format: ## Frame N (timing) • X words
    pattern = r'## Frame (\d+) \(([^)]+)\) [•·] (\d+) words\n\n(.*?)(?=\n## Frame|\n---|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        frame_num, timing, word_count, text = match
        # Remove [Visual: ...] annotations (match to end since they always appear last)
        clean_text = re.sub(r'\[Visual:.*$', '', text, flags=re.DOTALL).strip()
        # Clean up extra whitespace
        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
        frames.append({
            'number': int(frame_num),
            'timing': timing,
            'words': int(word_count),
            'transcript': clean_text
        })

    return video_title, frames


def wrap_text(text: str, canvas_obj, font: str, size: float, max_width: float) -> list:
    """Wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current_line = []

    canvas_obj.setFont(font, size)

    for word in words:
        test_line = ' '.join(current_line + [word])
        if canvas_obj.stringWidth(test_line, font, size) < max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def create_slides_pdf(video_dir: Path, frames_data: list, video_title: str, temp_dir: Path) -> Path:
    """
    Generate slides.pdf - frames only, full-page images.

    Args:
        video_dir: Path to video directory
        frames_data: List of frame dicts from parse_script
        video_title: Title of the video
        temp_dir: Path to temp directory for compressed images

    Returns:
        Path to generated PDF
    """
    frames_dir = video_dir / "frames"
    output_path = video_dir / "slides.pdf"

    # Create PDF with custom page size matching frame aspect ratio
    # Use 1920x1080 scaled to fit A4 landscape
    c = canvas.Canvas(str(output_path), pagesize=landscape(A4))

    for frame_data in frames_data:
        frame_num = frame_data['number']
        frame_path = frames_dir / f"frame_{frame_num}.png"

        if not frame_path.exists():
            print(f"  Warning: {frame_path.name} not found, skipping")
            continue

        # Compress image for smaller PDF
        compressed_path = compress_image(frame_path, temp_dir)

        # Get compressed image dimensions
        with Image.open(compressed_path) as img:
            img_width, img_height = img.size

        # Scale to fit page while maintaining aspect ratio
        page_w, page_h = PAGE_WIDTH, PAGE_HEIGHT
        scale = min(page_w / img_width, page_h / img_height)
        scaled_width = img_width * scale
        scaled_height = img_height * scale

        # Center on page
        x = (page_w - scaled_width) / 2
        y = (page_h - scaled_height) / 2

        c.drawImage(str(compressed_path), x, y, scaled_width, scaled_height)
        c.showPage()

    c.save()
    return output_path


def create_study_guide_pdf(video_dir: Path, frames_data: list, video_title: str, temp_dir: Path) -> Path:
    """
    Generate study_guide.pdf - frames + transcript text.

    Args:
        video_dir: Path to video directory
        frames_data: List of frame dicts from parse_script
        video_title: Title of the video
        temp_dir: Path to temp directory for compressed images

    Returns:
        Path to generated PDF
    """
    frames_dir = video_dir / "frames"
    output_path = video_dir / "study_guide.pdf"
    total_frames = len(frames_data)

    c = canvas.Canvas(str(output_path), pagesize=landscape(A4))

    for frame_data in frames_data:
        frame_num = frame_data['number']
        timing = frame_data['timing']
        transcript = frame_data['transcript']
        frame_path = frames_dir / f"frame_{frame_num}.png"

        if not frame_path.exists():
            print(f"  Warning: {frame_path.name} not found, skipping")
            continue

        # Compress image for smaller PDF
        compressed_path = compress_image(frame_path, temp_dir)

        # --- Header ---
        c.setFillColor(HEADER_COLOR)
        c.setFont("Helvetica-Bold", 14)
        header_y = PAGE_HEIGHT - HEADER_HEIGHT + 5*mm
        header_text = f"Frame {frame_num} ({timing})"
        c.drawString(MARGIN, header_y, header_text)

        # Page number (right aligned)
        page_num = f"{frame_num}/{total_frames}"
        c.drawRightString(PAGE_WIDTH - MARGIN, header_y, page_num)

        # --- Frame Image ---
        with Image.open(compressed_path) as img:
            img_width, img_height = img.size

        # Scale to fit width while maintaining aspect ratio
        available_width = PAGE_WIDTH - (2 * MARGIN)
        scale = available_width / img_width
        scaled_width = available_width
        scaled_height = img_height * scale

        # Cap height if too tall
        max_img_height = IMAGE_HEIGHT - 10*mm
        if scaled_height > max_img_height:
            scale = max_img_height / img_height
            scaled_height = max_img_height
            scaled_width = img_width * scale

        img_x = MARGIN + (available_width - scaled_width) / 2
        img_y = PAGE_HEIGHT - HEADER_HEIGHT - scaled_height - 5*mm

        c.drawImage(str(compressed_path), img_x, img_y, scaled_width, scaled_height)

        # --- Transcript Text ---
        c.setFillColor(TEXT_COLOR)
        text_y = img_y - 15*mm
        c.setFont("Helvetica", 13)  # Larger font (was 11)

        # Wrap text
        text_width = PAGE_WIDTH - (2 * MARGIN)
        lines = wrap_text(transcript, c, "Helvetica", 13, text_width)

        line_height = 16  # points (was 14)
        for line in lines:
            if text_y < FOOTER_HEIGHT + 10*mm:
                break  # Don't overflow into footer
            c.drawString(MARGIN, text_y, line)
            text_y -= line_height

        # --- Footer ---
        c.setFillColor(FOOTER_COLOR)
        c.setFont("Helvetica", 9)
        c.drawString(MARGIN, FOOTER_HEIGHT, video_title)

        c.showPage()

    c.save()
    return output_path


def generate_study_guides(video_dir: Path) -> tuple:
    """
    Generate both study guide PDFs for a video.

    Args:
        video_dir: Path to video directory (e.g., pipeline/LECTURE/Video-1)

    Returns:
        (slides_path, study_guide_path)
    """
    video_dir = Path(video_dir)
    frames_dir = video_dir / "frames"
    script_path = video_dir / "script.md"

    # Validate inputs
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Parse script
    video_title, frames_data = parse_script(script_path)

    if not frames_data:
        raise ValueError("No frames found in script.md")

    print(f"  Generating PDFs for: {video_title}")
    print(f"  Frames: {len(frames_data)}")

    # Use temp directory for compressed images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Generate slides.pdf (frames only)
        slides_path = create_slides_pdf(video_dir, frames_data, video_title, temp_path)
        print(f"  ✓ Created: {slides_path.name}")

        # Generate study_guide.pdf (frames + transcript)
        guide_path = create_study_guide_pdf(video_dir, frames_data, video_title, temp_path)
        print(f"  ✓ Created: {guide_path.name}")

    return slides_path, guide_path


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_study_guide.py <video_dir>")
        print()
        print("Generates two PDF study guides from video frames and script:")
        print("  - slides.pdf      : Frames only (slideshow)")
        print("  - study_guide.pdf : Frames + transcript text")
        print()
        print("Example:")
        print("  python generate_study_guide.py pipeline/BANK5016_Week1/Video-1")
        sys.exit(1)

    video_dir = Path(sys.argv[1])

    if not video_dir.exists():
        print(f"Error: Directory not found: {video_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Study Guide Generation")
    print("=" * 60)

    try:
        slides_path, guide_path = generate_study_guides(video_dir)
        print()
        print("✓ Study guides generated successfully")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
