#!/usr/bin/env python3
"""
TGIF-QA Count Dataset Splitter: Repetition (answer > 0) vs Non-Repetition (answer == 0)

This script splits TGIF-QA count question CSV/TSV files into two separate files:
  1. Non-repetition questions: where answer == 0 (action does not occur / 0 repetitions)
  2. Repetition questions: where answer > 0 (action repeated 1+ times)

Can be executed on any remote server using Python 3 (works with or without pandas).
"""

import os
import sys
import csv
import argparse
from collections import Counter
from typing import Optional, Tuple, Dict, Any


def detect_delimiter(file_path: str, default: str = '\t') -> str:
    """Detects delimiter (tab or comma) from the first line of the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if '\t' in first_line:
            return '\t'
        elif ',' in first_line:
            return ','
    return default


def split_dataset(
    input_path: str,
    output_zero_path: str,
    output_nonzero_path: str,
    delimiter: Optional[str] = None,
    export_gif_lists: bool = True,
    zero_gif_list_path: Optional[str] = None,
    nonzero_gif_list_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Splits the count QA dataset into zero and non-zero answer datasets.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if delimiter is None:
        delimiter = detect_delimiter(input_path)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(output_zero_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_nonzero_path)), exist_ok=True)

    header = None
    zero_rows = []
    nonzero_rows = []
    zero_gifs = set()
    nonzero_gifs = set()
    answer_dist = Counter()

    with open(input_path, 'r', encoding='utf-8') as f_in:
        reader = csv.reader(f_in, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"File {input_path} is empty.")

        # Find column indices
        col_names = [col.strip().lower() for col in header]
        
        if 'answer' in col_names:
            answer_idx = col_names.index('answer')
        else:
            # Fallback to column index 2 based on TGIF-QA format: gif_name, question, answer, ...
            answer_idx = 2
            
        gif_idx = col_names.index('gif_name') if 'gif_name' in col_names else 0

        for line_num, row in enumerate(reader, start=2):
            if not row or all(c.strip() == '' for c in row):
                continue  # skip blank lines

            if len(row) <= answer_idx:
                print(f"Warning [Line {line_num}]: Malformed row with {len(row)} columns (expected at least {answer_idx + 1}). Skipping.")
                continue

            raw_answer = row[answer_idx].strip()
            try:
                # Handle possible float representations e.g., '0.0' or '0'
                ans_val = int(float(raw_answer))
            except ValueError:
                print(f"Warning [Line {line_num}]: Could not parse answer '{raw_answer}' as integer. Storing in non-zero by default.")
                ans_val = -1

            answer_dist[ans_val] += 1
            gif_name = row[gif_idx].strip() if len(row) > gif_idx else ""

            if ans_val == 0:
                zero_rows.append(row)
                if gif_name:
                    zero_gifs.add(gif_name)
            else:
                nonzero_rows.append(row)
                if gif_name:
                    nonzero_gifs.add(gif_name)

    # Write zero answer file
    with open(output_zero_path, 'w', newline='', encoding='utf-8') as f_zero:
        writer = csv.writer(f_zero, delimiter=delimiter)
        writer.writerow(header)
        writer.writerows(zero_rows)

    # Write non-zero answer file
    with open(output_nonzero_path, 'w', newline='', encoding='utf-8') as f_nonzero:
        writer = csv.writer(f_nonzero, delimiter=delimiter)
        writer.writerow(header)
        writer.writerows(nonzero_rows)

    # Export text lists of GIF names if requested
    if export_gif_lists:
        if zero_gif_list_path:
            with open(zero_gif_list_path, 'w', encoding='utf-8') as f_zg:
                for gif in sorted(zero_gifs):
                    f_zg.write(f"{gif}\n")

        if nonzero_gif_list_path:
            with open(nonzero_gif_list_path, 'w', encoding='utf-8') as f_nzg:
                for gif in sorted(nonzero_gifs):
                    f_nzg.write(f"{gif}\n")

    stats = {
        'total_rows': len(zero_rows) + len(nonzero_rows),
        'zero_count': len(zero_rows),
        'nonzero_count': len(nonzero_rows),
        'unique_zero_gifs': len(zero_gifs),
        'unique_nonzero_gifs': len(nonzero_gifs),
        'overlap_gifs': len(zero_gifs.intersection(nonzero_gifs)),
        'answer_distribution': dict(sorted(answer_dist.items())),
        'delimiter': repr(delimiter)
    }

    return stats


def print_stats(stats: Dict[str, Any], output_zero: str, output_nonzero: str,
                zero_gif_list: Optional[str] = None, nonzero_gif_list: Optional[str] = None):
    print("=" * 60)
    print("           TGIF-QA COUNT SPLIT SUMMARY")
    print("=" * 60)
    print(f"Delimiter used:          {stats['delimiter']}")
    print(f"Total question samples:  {stats['total_rows']:,}")
    print(f"Zero / Non-repetition:   {stats['zero_count']:,} ({stats['zero_count'] / max(1, stats['total_rows']) * 100:.2f}%)")
    print(f"Non-zero / Repetition:   {stats['nonzero_count']:,} ({stats['nonzero_count'] / max(1, stats['total_rows']) * 100:.2f}%)")
    print("-" * 60)
    print(f"Unique Zero GIFs:        {stats['unique_zero_gifs']:,}")
    print(f"Unique Non-zero GIFs:    {stats['unique_nonzero_gifs']:,}")
    print(f"GIF Overlap between sets:{stats['overlap_gifs']:,}")
    print("-" * 60)
    print("Answer Value Distribution:")
    for ans, count in stats['answer_distribution'].items():
        print(f"  Count = {ans:<4} : {count:>7,} questions")
    print("-" * 60)
    print("Output Files Created:")
    print(f"  [Zero / Non-Repetition]: {output_zero}")
    print(f"  [Non-Zero / Repetition]: {output_nonzero}")
    if zero_gif_list and os.path.exists(zero_gif_list):
        print(f"  [Zero GIF List txt]:     {zero_gif_list}")
    if nonzero_gif_list and os.path.exists(nonzero_gif_list):
        print(f"  [Non-Zero GIF List txt]: {nonzero_gif_list}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Split TGIF-QA Count Dataset into 0-repetition and non-0 repetition files."
    )
    parser.add_argument(
        "--input", "-i",
        default="Total_count_question.csv",
        help="Path to input count QA file (default: Total_count_question.csv)"
    )
    parser.add_argument(
        "--output_dir", "-d",
        default=".",
        help="Directory to save output files (default: current directory)"
    )
    parser.add_argument(
        "--output_zero",
        default=None,
        help="Filename or path for zero-count questions (default: <input_base>_zero.csv)"
    )
    parser.add_argument(
        "--output_nonzero",
        default=None,
        help="Filename or path for non-zero count questions (default: <input_base>_nonzero.csv)"
    )
    parser.add_argument(
        "--sep", "--delimiter",
        default=None,
        help="Field delimiter ('\\t', ',', etc.). If omitted, automatically detected."
    )
    parser.add_argument(
        "--export_gif_lists",
        action="store_true",
        default=True,
        help="Also export plain-text lists of gif names for each split (default: True)"
    )
    parser.add_argument(
        "--no_gif_lists",
        action="store_false",
        dest="export_gif_lists",
        help="Disable exporting plain-text gif name lists"
    )

    args = parser.parse_args()

    input_path = args.input
    input_dir = os.path.dirname(os.path.abspath(input_path))
    input_base = os.path.splitext(os.path.basename(input_path))[0]
    input_ext = os.path.splitext(input_path)[1] or ".csv"

    out_dir = args.output_dir if args.output_dir != "." else input_dir

    if args.output_zero:
        output_zero_path = args.output_zero
    else:
        output_zero_path = os.path.join(out_dir, f"{input_base}_zero{input_ext}")

    if args.output_nonzero:
        output_nonzero_path = args.output_nonzero
    else:
        output_nonzero_path = os.path.join(out_dir, f"{input_base}_nonzero{input_ext}")

    zero_gif_list_path = os.path.join(out_dir, f"{input_base}_zero_gifs.txt") if args.export_gif_lists else None
    nonzero_gif_list_path = os.path.join(out_dir, f"{input_base}_nonzero_gifs.txt") if args.export_gif_lists else None

    # Handle escaped delimiter string if provided via command line e.g., '\t'
    delimiter = args.sep
    if delimiter == '\\t':
        delimiter = '\t'

    stats = split_dataset(
        input_path=input_path,
        output_zero_path=output_zero_path,
        output_nonzero_path=output_nonzero_path,
        delimiter=delimiter,
        export_gif_lists=args.export_gif_lists,
        zero_gif_list_path=zero_gif_list_path,
        nonzero_gif_list_path=nonzero_gif_list_path
    )

    print_stats(
        stats=stats,
        output_zero=output_zero_path,
        output_nonzero=output_nonzero_path,
        zero_gif_list=zero_gif_list_path,
        nonzero_gif_list=nonzero_gif_list_path
    )


if __name__ == "__main__":
    main()
