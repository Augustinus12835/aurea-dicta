#!/usr/bin/env python3
"""
Data Chart Generation for Aurea Dicta

Reads visual_specs.json and generates ONLY data charts:
- type="data_chart" → Generate Python code → Execute → PNG
- type="conceptual_diagram" → Skip (Gemini will create)

Uses Claude API to generate matplotlib code, then executes it.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.claude_client import ClaudeClient


DATA_CHART_PROMPT = """Write Python code to generate a data chart with HAND-DRAWN EDUCATIONAL STYLE.

CHART SPECIFICATION:
{spec}

REQUIREMENTS:
1. Use matplotlib for plotting
2. Data source: {data_source}
   - yfinance: `yf.download(ticker, start=start, end=end)`
   - FRED: `Fred(api_key=os.environ['FRED_API_KEY']).get_series(series_id, observation_start=start, observation_end=end)`
3. Output: Save PNG to: {output_path}
4. Resolution: figsize=(16, 9), dpi=120 → 1920x1080

CRITICAL: CLEAN EDUCATIONAL STYLE (No xkcd - use bold, clear styling)
DO NOT use plt.xkcd() - it causes font errors. Instead, use clean bold styling:
```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
plt.style.use('seaborn-v0_8-whitegrid')  # Clean white background with grid
```

STYLING GUIDELINES (Student-Friendly, Easy to Read):
- Background: WHITE (#FFFFFF) - clean, not cream/off-white
- Title: 24pt bold, fontweight='bold', pad=25, color='#1F2937'
- Axis labels: 18pt bold, fontweight='bold', color='#374151'
- Tick labels: 14pt, color='#4B5563'
- Grid: alpha=0.2, linestyle='--', color='#9CA3AF'
- Line width: 3-4pt (BOLD lines, easy to see)
- Spines: Hide top and right (ax.spines['top'].set_visible(False))

FONT SIZES (LARGE for video readability):
- plt.rcParams['font.size'] = 14
- plt.rcParams['axes.titlesize'] = 24
- plt.rcParams['axes.labelsize'] = 18
- plt.rcParams['xtick.labelsize'] = 14
- plt.rcParams['ytick.labelsize'] = 14
- plt.rcParams['legend.fontsize'] = 16

COLOR PALETTE (Bold, Distinct):
- Primary: '#E63946' (red), '#2A9D8F' (teal), '#F77F00' (orange), '#457B9D' (blue)
- For single series: '#2A9D8F' (teal) as default
- For crisis/negative: '#E63946' (red)
- For positive/growth: '#2A9D8F' (teal)
- For neutral/reference: '#457B9D' (blue)

ANNOTATION STYLES (Big, Clear Labels):
- "vertical_line_with_label" or "vertical_line_with_callout":
  - axvline with linewidth=2, linestyle='--'
  - text annotation with fontsize=14, fontweight='bold', bbox with white background
- "horizontal_line": axhline with linewidth=2, label in legend
- "horizontal_line_dashed": axhline with linestyle='--', linewidth=2
- "point_label": annotate with fontsize=14, arrow with arrowstyle='->', connectionstyle='arc3'
- "shaded_region": axvspan with alpha=0.2
- "horizontal_band": axhspan with alpha=0.15

EDUCATIONAL CLARITY RULES:
- Maximum 3-4 data series per chart (avoid clutter)
- Round numbers in labels (3.14159% → 3.1%)
- Clear, simple axis labels (no jargon)
- Annotate key events/points with callout boxes
- Use bbox=dict(boxstyle='round', facecolor='white', alpha=0.8) for text backgrounds

IMPORTANT CODE STRUCTURE:
- Include ALL necessary imports at the top (matplotlib, numpy, pandas, yfinance/fredapi, os, datetime)
- Set matplotlib backend: matplotlib.use('Agg')
- Set style: plt.style.use('seaborn-v0_8-whitegrid')
- Set rcParams for large fonts BEFORE creating figure
- Handle data fetching errors with try/except
- Create figure with: fig, ax = plt.subplots(figsize=(16, 9))
- Use tight_layout() before saving
- Save with: plt.savefig(output_path, dpi=120, facecolor='white', bbox_inches='tight')
- Close plot: plt.close()
- Print success message at the end

Write COMPLETE, EXECUTABLE Python code:

```python
"""


def load_visual_specs(video_dir: Path) -> Optional[Dict]:
    """Load visual_specs.json from video directory."""
    specs_path = video_dir / "visual_specs.json"
    if not specs_path.exists():
        return None
    with open(specs_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_data_charts(specs: Dict) -> List[Dict]:
    """Filter for data charts only (skip conceptual diagrams)."""
    visuals = specs.get("visuals", [])
    return [v for v in visuals if v.get("type") == "data_chart"]


def extract_code_block(response: str) -> str:
    """Extract Python code from LLM response."""
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        code = response.split("```")[1].split("```")[0]
    else:
        code = response
    return code.strip()


def generate_chart_code(
    client: ClaudeClient,
    spec: Dict,
    output_path: str,
    error_context: str = None
) -> str:
    """
    Generate Python code for a data chart using Claude.

    Args:
        client: Claude client instance
        spec: Chart specification from visual_specs.json
        output_path: Where to save the PNG
        error_context: Previous error message for retry

    Returns:
        Generated Python code
    """
    data_source = spec.get("data_source", "yfinance")

    prompt = DATA_CHART_PROMPT.format(
        spec=json.dumps(spec, indent=2),
        data_source=data_source,
        output_path=output_path
    )

    if error_context:
        prompt += f"\n\nPREVIOUS ATTEMPT FAILED WITH ERROR:\n{error_context}\n\nPlease fix the code to avoid this error."

    response = client.generate(
        prompt=prompt,
        system="You are an expert Python developer specializing in data visualization. Write clean, working matplotlib code. Always include all imports and error handling.",
        max_tokens=3000,
        temperature=0.2
    )

    return extract_code_block(response)


def execute_code(
    code: str,
    output_path: str,
    verbose: bool = False,
    timeout: int = 90
) -> Tuple[bool, str]:
    """
    Execute chart generation code.

    Args:
        code: Python code to execute
        output_path: Expected output path
        verbose: Show stdout/stderr
        timeout: Execution timeout in seconds

    Returns:
        (success, error_message)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ}
        )

        if verbose:
            if result.stdout:
                print(f"        stdout: {result.stdout[:500]}")
            if result.stderr:
                print(f"        stderr: {result.stderr[:500]}")

        if Path(output_path).exists():
            return True, ""
        else:
            error_msg = result.stderr or result.stdout or "Unknown error - file not created"
            return False, error_msg[:1000]

    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(temp_path)


def validate_output(output_path: Path) -> Dict:
    """
    Validate generated PNG file.

    Returns:
        Dict with validation results
    """
    if not output_path.exists():
        return {"valid": False, "error": "File does not exist"}

    size = output_path.stat().st_size

    if size < 10000:  # Less than 10KB is suspicious
        return {"valid": False, "error": f"File too small ({size} bytes)"}

    if size > 10_000_000:  # More than 10MB is suspicious
        return {"valid": False, "error": f"File too large ({size / 1_000_000:.1f} MB)"}

    return {"valid": True, "size_kb": size / 1024}


def process_chart(
    client: ClaudeClient,
    chart: Dict,
    diagrams_dir: Path,
    verbose: bool = False,
    dry_run: bool = False,
    max_retries: int = 0,
    force: bool = False
) -> Dict:
    """
    Process a single data chart specification.

    Args:
        client: Claude client
        chart: Chart specification
        diagrams_dir: Output directory
        verbose: Show detailed output
        dry_run: Generate code but don't execute
        max_retries: Number of retries on failure (default: 0, one-shot only)
        force: Regenerate even if output exists

    Returns:
        Dict with processing results
    """
    chart_id = chart.get("id", "unknown")
    chart_name = chart.get("name", "Unnamed")
    data_source = chart.get("data_source", "unknown")

    result = {
        "id": chart_id,
        "name": chart_name,
        "success": False,
        "attempts": 0,
        "error": None,
        "skipped": False
    }

    output_path = diagrams_dir / f"{chart_id}.png"
    code_path = diagrams_dir / f"{chart_id}_code.py"

    print(f"\n  [{chart_id}] {chart_name}")
    print(f"      Source: {data_source}")

    # Check if already generated (checkpoint)
    if not force and output_path.exists():
        validation = validate_output(output_path)
        if validation["valid"]:
            print(f"      Already exists: {chart_id}.png ({validation['size_kb']:.1f} KB) - skipping")
            result["success"] = True
            result["skipped"] = True
            result["output_path"] = str(output_path)
            result["size_kb"] = validation["size_kb"]
            return result

    error_context = None

    for attempt in range(max_retries + 1):
        result["attempts"] = attempt + 1

        # Generate code
        if attempt == 0:
            print(f"      Generating code...")
        else:
            print(f"      Retry {attempt}: Regenerating code with error context...")

        try:
            code = generate_chart_code(client, chart, str(output_path), error_context)
        except Exception as e:
            result["error"] = f"Code generation failed: {e}"
            print(f"      Error: {result['error']}")
            return result

        # Save code for debugging
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        if dry_run:
            print(f"      [DRY RUN] Code saved: {code_path}")
            result["success"] = True
            result["code_path"] = str(code_path)
            return result

        # Execute
        print(f"      Executing...")
        success, error_msg = execute_code(code, str(output_path), verbose)

        if success:
            # Validate output
            validation = validate_output(output_path)
            if validation["valid"]:
                print(f"      Created: {chart_id}.png ({validation['size_kb']:.1f} KB)")
                result["success"] = True
                result["output_path"] = str(output_path)
                result["code_path"] = str(code_path)
                result["size_kb"] = validation["size_kb"]
                return result
            else:
                error_context = validation["error"]
                print(f"      Validation failed: {error_context}")
        else:
            error_context = error_msg
            print(f"      Execution failed: {error_msg[:200]}")

    result["error"] = error_context or "Unknown error"
    return result


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_data_charts.py <pipeline_dir> [--video N] [--verbose] [--dry-run] [--force]")
        print()
        print("Generates data charts from visual_specs.json specifications.")
        print("Only processes type='data_chart' visuals; skips conceptual diagrams.")
        print("Skips charts that already exist (resumable). Use --force to regenerate.")
        print()
        print("Options:")
        print("  --video N     Process specific video only")
        print("  --verbose     Show detailed output")
        print("  --dry-run     Generate code but don't execute")
        print("  --force       Regenerate charts even if they exist")
        print()
        print("Examples:")
        print("  python generate_data_charts.py pipeline/YOUR_LECTURE")
        print("  python generate_data_charts.py pipeline/YOUR_LECTURE --video 1 --verbose")
        sys.exit(1)

    pipeline_dir = sys.argv[1]
    specific_video = None
    verbose = "--verbose" in sys.argv
    dry_run = "--dry-run" in sys.argv
    force = "--force" in sys.argv

    if "--video" in sys.argv:
        idx = sys.argv.index("--video")
        if idx + 1 < len(sys.argv):
            specific_video = int(sys.argv[idx + 1])

    print("=" * 60)
    print("Data Chart Generation")
    print("=" * 60)
    print(f"Pipeline: {pipeline_dir}")
    if dry_run:
        print("Mode: DRY RUN (code only, no execution)")

    # Check FRED API key
    if not os.environ.get('FRED_API_KEY'):
        print("\nWarning: FRED_API_KEY not set. FRED-based charts will fail.")

    pipeline_path = Path(pipeline_dir)

    # Find video directories
    if specific_video:
        video_dirs = [pipeline_path / f"Video-{specific_video}"]
    else:
        video_dirs = sorted(pipeline_path.glob("Video-*"))

    if not video_dirs:
        print(f"No video directories found in {pipeline_dir}")
        sys.exit(1)

    # Initialize Claude client
    client = ClaudeClient()

    results = {"success": [], "failed": [], "skipped": []}

    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue

        video_name = video_dir.name
        print(f"\n{video_name}:")

        # Load specs
        specs = load_visual_specs(video_dir)
        if not specs:
            print(f"  No visual_specs.json found")
            continue

        # Filter for data charts only
        data_charts = filter_data_charts(specs)
        total_visuals = len(specs.get("visuals", []))
        conceptual_count = total_visuals - len(data_charts)

        print(f"  Data charts: {len(data_charts)}")
        print(f"  Conceptual diagrams: {conceptual_count} (skipped - generated by Gemini)")

        if not data_charts:
            print(f"  No data charts to generate")
            continue

        # Create diagrams directory
        diagrams_dir = video_dir / "diagrams"
        diagrams_dir.mkdir(exist_ok=True)

        # Process each chart
        for chart in data_charts:
            result = process_chart(
                client=client,
                chart=chart,
                diagrams_dir=diagrams_dir,
                verbose=verbose,
                dry_run=dry_run,
                force=force
            )

            full_id = f"{video_name}/{result['id']}"

            if result["success"]:
                if result.get("skipped"):
                    results["skipped"].append(full_id)
                else:
                    results["success"].append(full_id)
            else:
                results["failed"].append(full_id)

    # Summary
    print("\n" + "=" * 60)
    print("DATA CHART GENERATION COMPLETE")
    print("=" * 60)
    generated = len(results['success'])
    skipped = len(results['skipped'])
    failed = len(results['failed'])
    print(f"  Generated: {generated}")
    if skipped > 0:
        print(f"  Skipped:   {skipped} (already exist)")
    print(f"  Failed:    {failed}")

    if results["failed"]:
        print("\nFailed charts (check *_code.py for debugging):")
        for f in results["failed"]:
            print(f"  - {f}")
        print("\nFix the *_code.py files manually, then run: python <code_file>.py")
        print("Or delete the code file and re-run to regenerate.")

    if generated > 0 or (skipped > 0 and failed == 0):
        print(f"\nNext step: python scripts/generate_scripts.py {pipeline_dir}")


if __name__ == "__main__":
    main()
