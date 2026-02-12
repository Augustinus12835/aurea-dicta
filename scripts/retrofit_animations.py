#!/usr/bin/env python3
"""
Retrofit Manim animations onto older videos that were compiled before
the animation pipeline existed.

For each video in a lecture:
  1. Run generate_math_animation.py (select + generate + render animated frames)
  2. Recompile with compile_video.py (now uses paired concat for sync)
  3. Delete old subtitles.srt so they can be regenerated with corrected offsets

Usage:
    # Single lecture
    python scripts/retrofit_animations.py pipeline/Calculus_1_Lecture_15_Slope_of_a_Curve_Velocity_an

    # Specific video only
    python scripts/retrofit_animations.py pipeline/Calculus_1_Lecture_15_Slope_of_a_Curve_Velocity_an --video 2

    # All lectures that have math_verification.json but no animated frames
    python scripts/retrofit_animations.py --all

    # Dry run (show what would be done)
    python scripts/retrofit_animations.py --all --dry-run

    # Force re-generate animation code (even if .py exists)
    python scripts/retrofit_animations.py pipeline/LECTURE --force-animate

    # Skip subtitle regeneration
    python scripts/retrofit_animations.py pipeline/LECTURE --skip-subtitles
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent.parent / "pipeline"
SCRIPTS_DIR = Path(__file__).parent


def find_videos(lecture_path: str, specific_video: int = None) -> list:
    """Find all Video-N folders in a lecture directory."""
    videos = []
    for entry in sorted(Path(lecture_path).iterdir()):
        if entry.is_dir() and entry.name.startswith("Video-"):
            num = int(entry.name.split("-")[1])
            if specific_video is not None and num != specific_video:
                continue
            videos.append(entry)
    return videos


def needs_animation(video_path: Path) -> bool:
    """Check if a video has math_verification.json but no animated .mp4 frames."""
    mv = video_path / "math_verification.json"
    if not mv.exists():
        return False
    # Check if any frame_N.mp4 already exists
    frames_dir = video_path / "frames"
    if frames_dir.exists():
        for f in frames_dir.iterdir():
            if f.name.startswith("frame_") and f.suffix == ".mp4":
                return False
    return True


def find_all_lectures() -> list:
    """Find all lectures that need retrofit (have math_verification but no animations)."""
    lectures = []
    for lec in sorted(PIPELINE_DIR.iterdir()):
        if not lec.is_dir():
            continue
        videos = find_videos(str(lec))
        if any(needs_animation(v) for v in videos):
            lectures.append(lec)
    return lectures


def run_step(cmd: list, label: str, dry_run: bool = False) -> bool:
    """Run a subprocess step with live output. Returns True on success."""
    print(f"    {label}", flush=True)
    if dry_run:
        print(f"      [dry-run] {' '.join(str(c) for c in cmd)}")
        return True

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in process.stdout:
        print(f"      {line}", end="", flush=True)
    process.wait()

    if process.returncode != 0:
        print(f"      FAILED (exit {process.returncode})")
        return False
    return True


def process_video(
    video_path: Path,
    dry_run: bool = False,
    force_animate: bool = False,
    skip_subtitles: bool = False,
) -> dict:
    """
    Process a single video: animate → recompile → delete subtitles.

    Returns dict with results.
    """
    video_name = f"{video_path.parent.name}/{video_path.name}"
    result = {"video": video_name, "animated": False, "compiled": False, "subtitles_cleared": False}

    mv_path = video_path / "math_verification.json"
    if not mv_path.exists():
        print(f"  {video_name}: No math_verification.json, skipping animation")
        # Still recompile (for paired concat fix) and clear subtitles
    else:
        # Step 1: Generate animations
        animate_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "generate_math_animation.py"),
            str(video_path),
        ]
        if force_animate:
            animate_cmd.append("--force")

        ok = run_step(animate_cmd, "[1/3] Generating animations...", dry_run)
        result["animated"] = ok
        if not ok:
            print(f"      Warning: Animation failed for {video_name}, continuing with recompile")

    # Step 2: Recompile
    compile_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "compile_video.py"),
        str(video_path),
    ]
    ok = run_step(compile_cmd, "[2/3] Recompiling video...", dry_run)
    result["compiled"] = ok

    # Step 3: Delete old subtitles
    if not skip_subtitles:
        srt_path = video_path / "subtitles.srt"
        if srt_path.exists():
            if dry_run:
                print(f"    [3/3] [dry-run] Would delete {srt_path}")
            else:
                srt_path.unlink()
                print(f"    [3/3] Deleted {srt_path.name}")
            result["subtitles_cleared"] = True
        else:
            print(f"    [3/3] No subtitles.srt to delete")
    else:
        print(f"    [3/3] Skipping subtitle cleanup (--skip-subtitles)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Retrofit Manim animations onto older videos"
    )
    parser.add_argument(
        "lecture_path",
        nargs="?",
        help="Path to lecture folder (e.g. pipeline/Calculus_1_Lecture_15...)",
    )
    parser.add_argument("--video", type=int, help="Process specific Video-N only")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all lectures that need animation",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument(
        "--force-animate",
        action="store_true",
        help="Regenerate animation code even if .py exists",
    )
    parser.add_argument(
        "--skip-subtitles",
        action="store_true",
        help="Don't delete old subtitles",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Skip animation, only recompile + clear subtitles (for sync fix only)",
    )

    args = parser.parse_args()

    if not args.lecture_path and not args.all:
        parser.error("Provide a lecture path or use --all")

    # Gather lectures to process
    if args.all:
        lectures = find_all_lectures()
        if not lectures:
            print("No lectures need retrofit.")
            return
        print(f"Found {len(lectures)} lectures needing animation:\n")
        for lec in lectures:
            print(f"  {lec.name}")
        print()
    else:
        lecture_path = args.lecture_path
        if not os.path.isabs(lecture_path):
            lecture_path = str(Path(__file__).parent.parent / lecture_path)
        if not os.path.exists(lecture_path):
            print(f"Error: {lecture_path} not found")
            sys.exit(1)
        lectures = [Path(lecture_path)]

    # Process
    all_results = []
    for lecture in lectures:
        print("=" * 70)
        print(f"LECTURE: {lecture.name}")
        print("=" * 70)

        videos = find_videos(str(lecture), args.video)
        if not videos:
            print("  No matching videos found.\n")
            continue

        for video_path in videos:
            print(f"\n  --- {video_path.name} ---")

            if args.compile_only:
                # Skip animation step entirely
                result = {"video": f"{lecture.name}/{video_path.name}"}
                compile_cmd = [
                    sys.executable,
                    str(SCRIPTS_DIR / "compile_video.py"),
                    str(video_path),
                ]
                ok = run_step(compile_cmd, "[1/2] Recompiling video...", args.dry_run)
                result["compiled"] = ok

                if not args.skip_subtitles:
                    srt_path = video_path / "subtitles.srt"
                    if srt_path.exists():
                        if args.dry_run:
                            print(f"    [2/2] [dry-run] Would delete {srt_path}")
                        else:
                            srt_path.unlink()
                            print(f"    [2/2] Deleted {srt_path.name}")
                        result["subtitles_cleared"] = True
                all_results.append(result)
            else:
                result = process_video(
                    video_path,
                    dry_run=args.dry_run,
                    force_animate=args.force_animate,
                    skip_subtitles=args.skip_subtitles,
                )
                all_results.append(result)

        print()

    # Summary
    print("=" * 70)
    print("RETROFIT SUMMARY")
    print("=" * 70)
    animated = sum(1 for r in all_results if r.get("animated"))
    compiled = sum(1 for r in all_results if r.get("compiled"))
    cleared = sum(1 for r in all_results if r.get("subtitles_cleared"))
    failed_compile = sum(1 for r in all_results if not r.get("compiled"))

    print(f"  Videos processed: {len(all_results)}")
    if not args.compile_only:
        print(f"  Animated: {animated}")
    print(f"  Recompiled: {compiled}")
    if failed_compile:
        print(f"  Failed to compile: {failed_compile}")
    print(f"  Subtitles cleared: {cleared}")

    if cleared:
        print(f"\n  To regenerate subtitles (resource-intensive, run separately):")
        for lecture in lectures:
            print(f"    python scripts/generate_subtitles.py {lecture}")

    if failed_compile:
        sys.exit(1)


if __name__ == "__main__":
    main()
