#!/usr/bin/env python3
"""
Extract CCA PowerOn rate schedules from San Diego Community Power PDF files and output YAML format.
"""

import argparse
import pathlib
import re
import sys
from typing import Dict

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


def parse_cca_rates(text: str) -> Dict:
    """Parse CCA PowerOn rates from PDF text.

    Returns dict mapping schedule_name to rates data.
    """
    results = {}
    lines = text.splitlines()

    # Pattern to match schedule names
    schedule_pattern = r'^\s*(DR-SES|DR-LI-MB|EV-TOU-\d+|EV-TOU|TOU-DR-\d+|TOU-DR|TOU-ELEC|DR|LS)\s+[A-Z]'

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for schedule header
        schedule_match = re.search(schedule_pattern, line)
        if schedule_match:
            schedule_name = schedule_match.group(1)
            schedule_data = {"summer": {}, "winter": {}}

            # Skip to data rows (look ahead for Season/Generation lines)
            i += 1
            while i < len(lines):
                data_line = lines[i]

                # Check if we've hit the next schedule
                if re.search(schedule_pattern, data_line):
                    i -= 1  # Back up so outer loop can process it
                    break

                # Parse data rows with rates
                # Format: "Season  ChargType  TOU_Period  $PowerOn  $PowerBase"
                # Look for lines with Summer/Winter and price values
                if ("Summer" in data_line or "Winter" in data_line) and "Generation" in data_line:
                    # Extract season
                    season = "summer" if "Summer" in data_line else "winter"

                    # Extract prices (PowerOn is first, PowerBase is second)
                    prices = re.findall(r'\$(\d+\.\d+)', data_line)

                    if len(prices) >= 2:
                        poweron_rate = float(prices[0])

                        # Determine TOU period type
                        if "On-Peak" in data_line and "Super Off-Peak" not in data_line:
                            schedule_data[season]["onpeak"] = poweron_rate
                        elif "Off-Peak" in data_line and "Super Off-Peak" not in data_line:
                            schedule_data[season]["offpeak"] = poweron_rate
                        elif "Super Off-Peak" in data_line:
                            schedule_data[season]["super_offpeak"] = poweron_rate
                        elif "Total" in data_line:
                            # Flat rate
                            schedule_data[season]["flat"] = poweron_rate

                i += 1

            # Only add if we found actual rate data
            has_data = any(
                len(schedule_data[season]) > 0 for season in ["summer", "winter"]
            )
            if has_data:
                results[schedule_name] = schedule_data

        i += 1

    return results


def process_pdf(pdf_path: pathlib.Path) -> Dict:
    """Process a single PDF file.
    Returns dict mapping schedule_name to data.
    """
    text = extract_text_with_layout(pdf_path)
    if not text:
        print(f"WARNING: Could not extract text from {pdf_path}")
        return {}

    return parse_cca_rates(text)


def main():
    output_dir = "rates"

    parser = argparse.ArgumentParser(
        description="Extract CCA PowerOn rate schedules from San Diego Community Power PDF files."
    )
    parser.add_argument("pdf_file", help="PDF file containing CCA rates")
    args = parser.parse_args()

    pdf_path = pathlib.Path(args.pdf_file)
    if not pdf_path.is_file():
        print(f"ERROR: {args.pdf_file} is not a file")
        sys.exit(1)

    print(f"Processing {pdf_path.name}...")
    cca_rates = process_pdf(pdf_path)

    if not cca_rates:
        print("No CCA rates extracted")
        sys.exit(1)

    # Write YAML
    import os
    output = os.path.join(output_dir, "cca_schedules.yaml")
    with open(output, "w") as f:
        yaml.dump(cca_rates, f, default_flow_style=False, sort_keys=False)

    print(f"Extracted {len(cca_rates)} CCA schedules to {output}")


if __name__ == "__main__":
    main()
