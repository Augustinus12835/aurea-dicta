#!/usr/bin/env python3
"""
YouTube Upload Script for Aurea Dicta

Uploads a video to YouTube with:
- AI-generated metadata (Claude API)
- AI-generated thumbnail (Gemini API)
- Timestamped captions from subtitles.srt

Usage:
    python scripts/upload_youtube.py pipeline/LECTURE/Video-1
    python scripts/upload_youtube.py pipeline/LECTURE/Video-1 --dry-run
    python scripts/upload_youtube.py pipeline/LECTURE/Video-1 --skip-thumbnail
    python scripts/upload_youtube.py pipeline/LECTURE/Video-1 --unlisted
    python scripts/upload_youtube.py pipeline/LECTURE/Video-1 --schedule "2025-01-15 18:00"
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

from scripts.utils.claude_client import ClaudeClient
from scripts.utils.youtube_client import YouTubeClient
import shutil


# Channel branding constants
CHANNEL_NAME = "Ludium"
CHANNEL_TAGLINE = "Learn. Play. Discover."
GITHUB_URL = "https://github.com/Augustinus12835/aurea-dicta"

# Playlist config location
PLAYLISTS_CONFIG_PATH = Path(__file__).parent.parent / "playlists.json"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload video to YouTube with AI-generated metadata"
    )
    parser.add_argument(
        "video_dir",
        type=str,
        help="Path to video directory (e.g., pipeline/LECTURE/Video-1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and generate metadata without uploading"
    )
    parser.add_argument(
        "--skip-thumbnail",
        action="store_true",
        help="Skip thumbnail generation and upload"
    )
    parser.add_argument(
        "--skip-captions",
        action="store_true",
        help="Skip caption/subtitle upload"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing upload_result.json"
    )
    parser.add_argument(
        "--regenerate-metadata",
        action="store_true",
        help="Regenerate metadata even if youtube_metadata.json exists"
    )
    parser.add_argument(
        "--unlisted",
        action="store_true",
        help="Upload video as unlisted instead of public"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        metavar="DATETIME",
        help="Schedule video release (Adelaide time). Format: 'YYYY-MM-DD HH:MM' e.g., '2025-01-15 18:00'"
    )
    parser.add_argument(
        "--playlist",
        type=str,
        nargs='?',
        const='auto',
        metavar="NAME",
        help="Add video to playlist. Use 'auto' or omit value to use playlists.json config, or specify explicit playlist name."
    )
    parser.add_argument(
        "--animated",
        action="store_true",
        help="Upload animated variant (final_video_animated.mp4 with thumbnail_animated.png)"
    )
    return parser.parse_args()


def validate_files(video_dir: Path, animated: bool = False) -> dict:
    """
    Validate required files exist in video directory.

    Returns:
        Dictionary with file paths and existence status
    """
    if animated:
        files = {
            "video": video_dir / "final_video_animated.mp4",
            "brief": video_dir / "video_brief.md",
            "content": video_dir / "content.txt",
            "subtitles": video_dir / "subtitles.srt",
            "thumbnail": video_dir / "thumbnail_animated.png",
            "metadata": video_dir / "youtube_metadata_animated.json",
            "upload_result": video_dir / "upload_result_animated.json",
        }
    else:
        files = {
            "video": video_dir / "final_video.mp4",
            "brief": video_dir / "video_brief.md",
            "content": video_dir / "content.txt",
            "subtitles": video_dir / "subtitles.srt",
            "thumbnail": video_dir / "thumbnail.png",
            "metadata": video_dir / "youtube_metadata.json",
            "upload_result": video_dir / "upload_result.json",
        }

    status = {}
    for name, path in files.items():
        status[name] = {
            "path": path,
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0
        }

    return status


def get_source_info(video_dir: Path) -> dict:
    """
    Extract original source information from the lecture directory.

    Returns:
        Dictionary with source URL, title, etc.
    """
    source_info = {
        "url": None,
        "title": None,
        "author": None
    }

    # Check parent directory for transcript.json (contains source info)
    parent_dir = video_dir.parent
    transcript_path = parent_dir / "transcript.json"

    if transcript_path.exists():
        try:
            with open(transcript_path) as f:
                transcript = json.load(f)

            if transcript.get("source") == "youtube":
                video_id = transcript.get("video_id")
                if video_id:
                    source_info["url"] = f"https://www.youtube.com/watch?v={video_id}"

            # Also check for source_info.json if it exists
            source_info_path = parent_dir / "source_info.json"
            if source_info_path.exists():
                with open(source_info_path) as f:
                    extra = json.load(f)
                    source_info.update(extra)

        except Exception as e:
            print(f"  Warning: Could not read source info: {e}")

    return source_info


def generate_youtube_metadata(video_dir: Path, source_info: dict = None) -> dict:
    """
    Generate YouTube metadata using Claude API.

    Args:
        video_dir: Path to video directory
        source_info: Optional source information

    Returns:
        Dictionary with title, description, tags
    """
    script_path = video_dir / "script.md"
    script = script_path.read_text() if script_path.exists() else ""

    if not script:
        raise ValueError(f"script.md not found in {video_dir}")

    # Build source credit section
    source_credit = ""
    if source_info and source_info.get("url"):
        source_credit = f"""
ORIGINAL SOURCE:
URL: {source_info.get('url', 'Unknown')}
Title: {source_info.get('title', 'Original lecture')}
Author: {source_info.get('author', 'Original creator')}

IMPORTANT: Include a prominent credit section for the original source in the description.
"""

    # Single style: journalistic precision, content-driven, never clickbait
    title_instructions = """1. **Title** - A clear, content-driven title with journalistic precision (max 70 characters)

   TITLE PHILOSOPHY:
   Think like a quality documentary or magazine editor. The title should be:
   - CONTENT-DRIVEN: What is this actually about? Lead with the concept.
   - CLEAR: A viewer should know exactly what they'll learn
   - PRECISE: Journalistic accuracy, no exaggeration
   - ENGAGING through substance, not manipulation

   GOOD TITLE PATTERNS:
   - Lead with the concept: "The Yield Curve and What It Tells Us About Recessions"
   - Highlight the insight: "Duration: The Single Number That Measures Bond Risk"
   - Ask the real question: "How Do Bond Prices Actually Respond to Interest Rates?"
   - State the core idea: "Credit Spreads and the Price of Default Risk"

   AVOID:
   - Clickbait hooks ("This Chart Predicted...", "What They Don't Want You to Know")
   - Sensationalism ("SHOCKING", "Mind-Blowing", "Game-Changing")
   - Vague curiosity gaps ("The Truth About...", "What Really Happened")
   - Overused YouTube patterns ("Explained", "Ultimate Guide", "Everything You Need")

   The title should work equally well as a chapter heading in a textbook or a headline in The Economist."""

    system_prompt = "You are an educational content editor with journalistic standards. Create titles that are clear, precise, and content-driven. Never use clickbait tactics. Your titles should read like quality documentary titles or magazine headlines - engaging through substance, not manipulation. Output valid JSON only."

    prompt = f"""Generate YouTube metadata for this educational video from the channel "{CHANNEL_NAME}".

SCRIPT (the actual narration - extract key concepts and specific details from this):
{script}

{source_credit}

CHANNEL INFO:
- Channel: {CHANNEL_NAME}
- Tagline: "{CHANNEL_TAGLINE}"
- GitHub: {GITHUB_URL}
- Focus: Distilling long lectures into focused concept videos

Generate:
{title_instructions}

2. **Description** - Full YouTube description including:
   - Hook/summary (2-3 sentences about what viewers will learn)
   - Section: "Key concepts covered:" with bullet points
   - Section: "ORIGINAL SOURCE" with full credit and link to original
   - Section: "About {CHANNEL_NAME}" with channel info
   - Relevant hashtags at the end (5-8 hashtags)
   - IMPORTANT: Do NOT use < or > characters anywhere in the description (YouTube API rejects them). Write "greater than" or "positive/negative" instead of mathematical inequalities.

3. **Tags** - List of 10-15 relevant keywords for YouTube search

OUTPUT FORMAT (JSON only, no markdown):
{{
    "title": "Your Video Title Here",
    "description": "Full description with sections...",
    "tags": ["tag1", "tag2", "tag3", ...]
}}"""

    client = ClaudeClient()
    response = client.generate(
        prompt=prompt,
        system=system_prompt,
        max_tokens=3000,
        temperature=0.5
    )

    # Parse JSON from response
    try:
        # Handle potential markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        metadata = json.loads(response.strip())
        return metadata

    except json.JSONDecodeError as e:
        print(f"  Error parsing Claude response: {e}")
        print(f"  Raw response: {response[:500]}...")
        raise


def copy_frame_as_thumbnail(video_dir: Path) -> Path:
    """
    Copy frame_0.png as the YouTube thumbnail, compressing if over 2MB.

    The title slide (frame 0) is already designed to be visually appealing
    and represents the video content well.

    Args:
        video_dir: Path to video directory

    Returns:
        Path to thumbnail, or None if frame_0 doesn't exist
    """
    from PIL import Image

    frame_0_path = video_dir / "frames" / "frame_0.png"
    thumbnail_path = video_dir / "thumbnail.png"

    if not frame_0_path.exists():
        return None

    max_bytes = 2 * 1024 * 1024  # YouTube's 2MB limit

    if frame_0_path.stat().st_size <= max_bytes:
        shutil.copy2(frame_0_path, thumbnail_path)
    else:
        img = Image.open(frame_0_path)
        # Save as JPEG with high quality to stay under 2MB
        quality = 95
        while quality >= 50:
            img.save(thumbnail_path, "JPEG", quality=quality)
            if thumbnail_path.stat().st_size <= max_bytes:
                break
            quality -= 5
        print(f"Thumbnail compressed to JPEG (quality={quality})")

    return thumbnail_path


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def parse_schedule_datetime(schedule_str: str) -> str:
    """
    Parse schedule datetime string and convert to UTC ISO 8601 format.

    Args:
        schedule_str: Datetime in Adelaide time, format 'YYYY-MM-DD HH:MM'

    Returns:
        ISO 8601 datetime string in UTC (e.g., '2025-01-15T07:30:00.0Z')

    Raises:
        ValueError: If datetime format is invalid
    """
    adelaide_tz = ZoneInfo("Australia/Adelaide")
    utc_tz = ZoneInfo("UTC")

    try:
        # Parse the datetime string
        local_dt = datetime.strptime(schedule_str, "%Y-%m-%d %H:%M")
        # Attach Adelaide timezone
        local_dt = local_dt.replace(tzinfo=adelaide_tz)
        # Convert to UTC
        utc_dt = local_dt.astimezone(utc_tz)
        # Format for YouTube API (ISO 8601)
        return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.0Z")
    except ValueError:
        raise ValueError(
            f"Invalid datetime format: '{schedule_str}'. "
            f"Expected format: 'YYYY-MM-DD HH:MM' (e.g., '2025-01-15 18:00')"
        )


def load_playlist_config(channel_type: str = "public") -> dict:
    """
    Load playlist configuration from playlists.json.

    Args:
        channel_type: "public" or "personal"

    Returns:
        Dictionary mapping folder prefixes to playlist names
    """
    if not PLAYLISTS_CONFIG_PATH.exists():
        return {}

    try:
        with open(PLAYLISTS_CONFIG_PATH) as f:
            config = json.load(f)
        return config.get(channel_type, {})
    except Exception as e:
        print(f"  Warning: Could not load playlist config: {e}")
        return {}


def get_playlist_name_for_folder(folder_name: str, channel_type: str = "public") -> str | None:
    """
    Get playlist name for a lecture folder using playlists.json config.

    Args:
        folder_name: Lecture folder name (e.g., "BANK5016_Week1")
        channel_type: "public" or "personal"

    Returns:
        Playlist name if matched, None if no match or explicitly set to null
    """
    config = load_playlist_config(channel_type)

    # Skip comment keys
    for prefix, playlist_name in config.items():
        if prefix.startswith("_"):
            continue

        # Wildcard fallback - auto-derive from folder name
        if prefix == "*":
            if playlist_name is None:
                return None
            # Auto-derive: replace underscores with spaces, clean up
            return folder_name.replace("_", " ").strip()

        # Prefix match
        if folder_name.startswith(prefix):
            return playlist_name

    return None


def main():
    args = parse_args()
    video_dir = Path(args.video_dir)

    # Header
    video_name = video_dir.name
    print()
    print(f"YouTube Upload - {video_name}")
    print("=" * 50)

    # Validate directory exists
    if not video_dir.exists():
        print(f"Error: Directory not found: {video_dir}")
        sys.exit(2)

    # Validate files
    files = validate_files(video_dir, animated=args.animated)

    # Required files
    video_filename = "final_video_animated.mp4" if args.animated else "final_video.mp4"
    if not files["video"]["exists"]:
        print(f"Error: {video_filename} not found in {video_dir}")
        if args.animated:
            print("Run the animated pipeline first: python scripts/pipeline.py animate LECTURE")
        else:
            print("Run the full pipeline first to generate the video.")
        sys.exit(2)

    if not files["brief"]["exists"]:
        print(f"Error: video_brief.md not found in {video_dir}")
        sys.exit(2)

    if not files["content"]["exists"]:
        print(f"Error: content.txt not found in {video_dir}")
        sys.exit(2)

    # Print file status
    print(f"Found: {video_filename} ({format_file_size(files['video']['size'])})")
    print(f"Found: video_brief.md")
    print(f"Found: content.txt")

    if files["subtitles"]["exists"]:
        print(f"Found: subtitles.srt")
    else:
        print(f"Warning: subtitles.srt not found - captions will be skipped")

    if files["thumbnail"]["exists"]:
        print(f"Found: thumbnail.png")
    else:
        print(f"Note: thumbnail.png not found - will generate")

    # Check for existing upload
    if files["upload_result"]["exists"] and not args.force:
        print()
        print(f"Video already uploaded (upload_result.json exists)")
        print(f"Use --force to upload again")
        sys.exit(0)

    # Get source info for attribution
    source_info = get_source_info(video_dir)
    if source_info.get("url"):
        print(f"Source: {source_info['url']}")

    print()

    # Generate or load metadata
    if files["metadata"]["exists"] and not args.regenerate_metadata:
        print(f"Loading existing {files['metadata']['path'].name}...")
        with open(files["metadata"]["path"]) as f:
            metadata = json.load(f)
        print("Loaded metadata from file")
    elif args.animated:
        # Animated: derive from static metadata (no API call needed)
        static_metadata_path = video_dir / "youtube_metadata.json"
        if not static_metadata_path.exists():
            print("Error: youtube_metadata.json not found — upload static version first")
            sys.exit(2)
        with open(static_metadata_path) as f:
            metadata = json.load(f)
        metadata["title"] = "[Animated] " + metadata["title"]
        metadata["description"] = (
            "Animated version with step-by-step math walkthroughs powered by Manim.\n\n"
            + metadata["description"]
        )
        metadata["tags"] = metadata.get("tags", []) + ["animated", "manim", "math animation"]
        # Save animated metadata
        with open(files["metadata"]["path"], "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Generated {files['metadata']['path'].name} (derived from static)")
    else:
        print("Generating metadata via Claude API...")
        metadata = generate_youtube_metadata(video_dir, source_info)

        # Save metadata
        with open(files["metadata"]["path"], "w") as f:
            json.dump(metadata, f, indent=2)
        print("Generated youtube_metadata.json")

    # Display metadata
    print()
    print(f"Title: {metadata.get('title', 'Unknown')}")
    print(f"Tags: {', '.join(metadata.get('tags', [])[:5])}...")
    print()

    # Dry run - stop here
    if args.dry_run:
        print("DRY RUN - Would upload with this metadata:")
        print()
        print("Description:")
        print("-" * 40)
        print(metadata.get("description", "No description")[:500])
        print("...")
        print("-" * 40)
        print()
        print("To actually upload, run without --dry-run")
        sys.exit(0)

    # Authenticate with YouTube
    print("Authenticating with YouTube...")
    youtube = YouTubeClient()
    if not youtube.authenticate():
        print("Error: YouTube authentication failed")
        sys.exit(2)

    channel_name = youtube.channel_info.get("title", "Unknown")
    print(f"Authenticated as: {channel_name}")
    print()

    # Determine privacy status and scheduling
    publish_at = None
    if args.schedule:
        if args.unlisted:
            print("Error: Cannot use --schedule with --unlisted")
            print("Scheduled videos must be private until release, then become public.")
            sys.exit(1)
        try:
            publish_at = parse_schedule_datetime(args.schedule)
            adelaide_tz = ZoneInfo("Australia/Adelaide")
            local_dt = datetime.strptime(args.schedule, "%Y-%m-%d %H:%M").replace(tzinfo=adelaide_tz)
            print(f"Scheduled release: {local_dt.strftime('%Y-%m-%d %H:%M')} Adelaide time")
            privacy_status = "private"  # Will be set to private, then public at publish_at
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.unlisted:
        privacy_status = "unlisted"
    else:
        privacy_status = "public"

    # Upload video
    status_msg = f"scheduled for {args.schedule}" if args.schedule else privacy_status
    print(f"Uploading video ({status_msg})...")
    video_id = youtube.upload_video(
        video_path=files["video"]["path"],
        title=metadata["title"],
        description=metadata["description"],
        tags=metadata.get("tags", []),
        category_id="27",  # Education
        privacy_status=privacy_status,
        default_language="en",
        default_audio_language="en",
        publish_at=publish_at,
        progress_callback=lambda p: print(f"  Progress: {p}%", end="\r")
    )

    if not video_id:
        print("Error: Video upload failed")
        sys.exit(1)

    video_url = youtube.get_video_url(video_id)
    print(f"Video uploaded: {video_url}")

    # Upload captions
    captions_uploaded = False
    if not args.skip_captions and files["subtitles"]["exists"]:
        print("Uploading captions...")
        captions_uploaded = youtube.upload_captions(
            video_id=video_id,
            srt_path=files["subtitles"]["path"],
            language="en",
            name="English"
        )
        if captions_uploaded:
            print("Captions uploaded")
        else:
            print("Warning: Caption upload failed")
    elif args.skip_captions:
        print("Skipping captions (--skip-captions)")
    else:
        print("Skipping captions (subtitles.srt not found)")

    # Generate thumbnail via Gemini and upload
    thumbnail_uploaded = False
    if not args.skip_thumbnail:
        # Generate thumbnail if it doesn't exist
        if not files["thumbnail"]["exists"]:
            if args.animated:
                # Generate animated thumbnail from static thumbnail
                static_thumb = video_dir / "thumbnail.png"
                if static_thumb.exists():
                    print("Generating animated thumbnail (badge overlay)...")
                    try:
                        from scripts.generate_animated_thumbnail import add_animated_badge
                        add_animated_badge(str(static_thumb), str(files["thumbnail"]["path"]))
                        print(f"Thumbnail generated ({files['thumbnail']['path'].stat().st_size / 1024:.0f} KB)")
                    except Exception as e:
                        print(f"Warning: Could not generate animated thumbnail: {e}")
                else:
                    print("Warning: No static thumbnail.png to derive animated thumbnail from")
            else:
                print("Generating thumbnail via Gemini API...")
                try:
                    from scripts.generate_thumbnail import generate_thumbnail, get_video_title
                    title = get_video_title(video_dir)
                    result = generate_thumbnail(video_dir, title)
                    if result:
                        print(f"Thumbnail generated ({result.stat().st_size / 1024:.0f} KB)")
                    else:
                        print("Warning: Thumbnail generation failed")
                except Exception as e:
                    print(f"Warning: Could not generate thumbnail: {e}")

        # Upload thumbnail
        if files["thumbnail"]["path"].exists():
            print("Uploading thumbnail...")
            thumbnail_uploaded = youtube.upload_thumbnail(
                video_id=video_id,
                thumbnail_path=files["thumbnail"]["path"]
            )
            if thumbnail_uploaded:
                print("Thumbnail uploaded")
            else:
                print("Warning: Thumbnail upload failed (account may need verification)")
    else:
        print("Skipping thumbnail (--skip-thumbnail)")

    # Add to playlist if requested
    playlist_id = None
    playlist_name = None
    if args.playlist:
        # Determine playlist name
        if args.playlist == 'auto':
            # Get lecture folder name (parent of Video-N directory)
            lecture_folder = video_dir.parent.name
            channel_type = "animated_public" if args.animated else "public"
            playlist_name = get_playlist_name_for_folder(lecture_folder, channel_type)
            if playlist_name:
                print(f"Matched playlist: {playlist_name}")
            else:
                print(f"No playlist match for folder: {lecture_folder}")
        else:
            playlist_name = args.playlist

        if playlist_name:
            print(f"Adding to playlist: {playlist_name}...")
            playlist_id = youtube.get_or_create_playlist(
                title=playlist_name,
                description=f"Videos from {CHANNEL_NAME}",
                privacy_status="public"
            )
            if playlist_id:
                if youtube.add_video_to_playlist(playlist_id, video_id):
                    playlist_url = youtube.get_playlist_url(playlist_id)
                    print(f"Added to playlist: {playlist_url}")
                else:
                    print("Warning: Failed to add video to playlist")
            else:
                print("Warning: Failed to get/create playlist")

    # Save upload result
    result = {
        "video_id": video_id,
        "video_url": video_url,
        "upload_timestamp": datetime.now().isoformat(),
        "title": metadata["title"],
        "status": "scheduled" if publish_at else privacy_status,
        "scheduled_publish_at": publish_at,
        "scheduled_publish_local": args.schedule if args.schedule else None,
        "captions_uploaded": captions_uploaded,
        "thumbnail_uploaded": thumbnail_uploaded,
        "channel_id": youtube.channel_info.get("id", ""),
        "channel_title": youtube.channel_info.get("title", ""),
        "playlist_id": playlist_id,
        "playlist_name": playlist_name
    }

    with open(files["upload_result"]["path"], "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 50)
    print(f"Upload complete!")
    print(f"URL: {video_url}")
    print(f"Saved: upload_result.json")
    print()


if __name__ == "__main__":
    main()
