#!/usr/bin/env python3
"""
Extract rate schedules from SDGE PDF files and output YAML format.
"""

import argparse
import pathlib
import re
import sys
from typing import Dict, List, Tuple
import os

import yaml

try:
    from pypdf import PdfReader
except ImportError:
    print("ERROR: pypdf is required. Install with: pip install pypdf")
    sys.exit(1)


def extract_text_with_layout(pdf_path: pathlib.Path) -> str:
    """Extract text from PDF preserving layout."""
    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        text += page.extract_text(extraction_mode="layout")
    return text


def parse_schedule_name(filename: str) -> str:
    """Extract schedule name from filename.
    Example: '1-1-26 Schedule DR Total Rates Table.pdf' -> 'DR'
    """
    # Pattern: Schedule <NAME> Total Rates Table
    match = re.search(r"Schedule\s+([A-Z0-9\-]+)\s+Total", filename, re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback: remove extension and split
    name = filename.rsplit(".", 1)[0]
    # Remove prefix up to 'Schedule'
    parts = name.split()
    try:
        idx = parts.index("Schedule")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return name


def extract_sections(text: str) -> List[Tuple[str, str]]:
    """Extract sections for each schedule in the PDF text.
    Returns list of (schedule_name, section_text)."""
    sections = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for "SCHEDULE " prefix (uppercase)
        if line.startswith("SCHEDULE "):
            # Extract schedule name: "SCHEDULE EV-TOU" or "SCHEDULE EV-TOU-2"
            # The name may have extra spaces, but we take the next token
            parts = line.split()
            if len(parts) >= 2:
                schedule_name = parts[1]
                # Gather section text from this line until next SCHEDULE or end
                section_lines = []
                j = i
                while j < len(lines) and (
                    j == i or not lines[j].strip().startswith("SCHEDULE ")
                ):
                    section_lines.append(lines[j])
                    j += 1
                section_text = "\n".join(section_lines)
                sections.append((schedule_name, section_text))
                i = j - 1  # will be incremented
        i += 1
    return sections


def determine_type(text: str) -> str:
    """Determine schedule type: tier, sop, or op.
    Rules: if contains tier -> tier, elif contains super_offpeak -> sop, else op."""
    if re.search(r"\bTier\s+[12]\b", text, re.IGNORECASE):
        return "tier"
    if re.search(r"Super Off-Peak", text, re.IGNORECASE):
        return "sop"
    # Default to op (offpeak/onpeak only)
    return "op"


def parse_rates(text: str, schedule_type: str) -> Dict:
    """Parse rates from PDF text."""
    # Split into lines
    lines = text.splitlines()
    # Find Summer and Winter sections
    summer_start = None
    winter_start = None
    for i, line in enumerate(lines):
        if "Summer" in line and summer_start is None:
            summer_start = i
        if "Winter" in line and winter_start is None:
            winter_start = i
    # If not found, try case-insensitive
    if summer_start is None:
        for i, line in enumerate(lines):
            if "summer" in line.lower():
                summer_start = i
                break
    if winter_start is None:
        for i, line in enumerate(lines):
            if "winter" in line.lower():
                winter_start = i
                break
    # Extract sections
    summer_text = (
        "\n".join(lines[summer_start:winter_start])
        if summer_start is not None and winter_start is not None
        else ""
    )
    winter_text = "\n".join(lines[winter_start:]) if winter_start is not None else ""

    # Parse each section
    summer_rates = parse_section(summer_text, schedule_type, "summer")
    winter_rates = parse_section(winter_text, schedule_type, "winter")

    return {"summer": summer_rates, "winter": winter_rates}


def parse_numeric_token(token: str) -> float:
    """Convert token to float, handling parentheses as negative."""
    token = token.strip()
    if token.startswith("(") and token.endswith(")"):
        return -float(token[1:-1])
    else:
        return float(token)


def parse_section(section_text: str, schedule_type: str, season: str) -> Dict:
    """Parse a season section."""
    lines = section_text.splitlines()
    rates = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for Baseline Adjustment Credit
        if "Baseline Adjustment Credit" in line:
            # Extract numeric tokens similarly
            tokens = line.split()
            numeric_tokens = []
            for token in tokens:
                if re.match(
                    r"^\(?\d+\.?\d*\)?$", token.replace("(", "").replace(")", "")
                ):
                    numeric_tokens.append(token)
            if len(numeric_tokens) >= 2:
                total_token = numeric_tokens[-1]
                eecc_token = numeric_tokens[-2]
                try:
                    total = parse_numeric_token(total_token)
                    eecc = parse_numeric_token(eecc_token)
                    # For baseline credit, we store the credit amount (negative)
                    # The credit appears to be the total column
                    rates["baseline_adjustment_credit"] = round(total, 5)
                except ValueError:
                    pass
            continue

        # Determine subtype based on schedule type
        subtype = None
        if schedule_type == "tier":
            if "Tier 1" in line:
                subtype = "tier1"
            elif "Tier 2" in line:
                subtype = "tier2"
        elif schedule_type == "sop":
            # Check for TOU subtypes with Super Off-Peak
            line_stripped = line.lstrip()
            if line_stripped.startswith("Super Off-Peak"):
                subtype = "super_offpeak"
            elif line_stripped.startswith("Off-Peak"):
                subtype = "offpeak"
            elif line_stripped.startswith("On-Peak"):
                subtype = "onpeak"
        else:  # op
            # Only Off-Peak and On-Peak
            line_stripped = line.lstrip()
            if line_stripped.startswith("Off-Peak"):
                subtype = "offpeak"
            elif line_stripped.startswith("On-Peak"):
                subtype = "onpeak"

        if not subtype:
            continue

        # Split line into tokens
        tokens = line.split()
        # Extract numeric tokens (including parentheses)
        numeric_tokens = []
        for token in tokens:
            # Check if token looks like a number (with optional parentheses)
            if re.match(r"^\(?\d+\.?\d*\)?$", token.replace("(", "").replace(")", "")):
                numeric_tokens.append(token)

        # Need at least two numeric tokens (EECC and Total)
        if len(numeric_tokens) < 2:
            continue

        # Last numeric token is Total, second last is EECC
        total_token = numeric_tokens[-1]
        eecc_token = numeric_tokens[-2]

        try:
            total = parse_numeric_token(total_token)
            eecc = parse_numeric_token(eecc_token)
            tariffs = total - eecc
            rates[subtype] = {"tariffs": round(tariffs, 5), "eecc": round(eecc, 5)}
        except ValueError:
            continue

    return rates


def process_pdf(pdf_path: pathlib.Path) -> Dict:
    """Process a single PDF file, possibly containing multiple schedules.
    Returns dict mapping schedule_name to data."""
    text = extract_text_with_layout(pdf_path)
    if not text:
        print(f"WARNING: Could not extract text from {pdf_path}")
        return {}

    results = {}
    sections = extract_sections(text)
    if sections:
        for schedule_name, section_text in sections:
            schedule_type = determine_type(section_text)
            rates = parse_rates(section_text, schedule_type)
            results[schedule_name] = {
                "type": schedule_type,
                "summer": rates["summer"],
                "winter": rates["winter"],
            }
    else:
        # fallback to filename-based schedule name
        schedule_name = parse_schedule_name(pdf_path.name)
        schedule_type = determine_type(text)
        rates = parse_rates(text, schedule_type)
        results[schedule_name] = {
            "type": schedule_type,
            "summer": rates["summer"],
            "winter": rates["winter"],
        }

    return results


def main():
    output_dir = "schedules"

    parser = argparse.ArgumentParser(
        description="Extract rate schedules from SDGE PDF files."
    )
    parser.add_argument("directory", help="Directory containing PDF files")
    args = parser.parse_args()

    pdf_dir = pathlib.Path(args.directory)
    if not pdf_dir.is_dir():
        print(f"ERROR: {args.directory} is not a directory")
        sys.exit(1)

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {args.directory}")
        sys.exit(0)

    all_rates = {}
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        result = process_pdf(pdf_file)
        if result:
            all_rates.update(result)

    # Write YAML

    output = os.path.join(output_dir, "sdge_schedules.yaml")
    with open(output, "w") as f:
        yaml.dump(all_rates, f, default_flow_style=False, sort_keys=False)

    print(f"Extracted {len(all_rates)} schedules to {output}")


if __name__ == "__main__":
    main()
