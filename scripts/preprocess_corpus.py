#!/usr/bin/env python3
"""
Script to combine and preprocess JSONL corpus files.
Features:
- Combines multiple JSONL files from a directory (recursive).
- Deduplicates documents based on exact content match.
- Deduplicates content based on sliding window or paragraph hashing (to remove repeated boilerplate).
- Filters by minimum length.
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Set, List, Dict, Generator

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_content_hash(text: str) -> str:
    """Returns MD5 hash of the text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def normalize_text(text: str) -> str:
    """Simple text normalization."""
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_jsonl_files(input_dir: Path) -> Generator[Dict, None, None]:
    """Recursively yields JSON objects from .jsonl files in input_dir."""
    files = sorted(list(input_dir.rglob("*.jsonl")))
    logger.info(f"Found {len(files)} JSONL files in {input_dir}")
    
    for file_path in tqdm(files, desc="Reading files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield data
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON in {file_path} at line {line_num}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")

def preprocess_and_combine(
    input_dir: Path,
    output_file: Path,
    min_length: int,
    dedup_mode: str,
    sequence_min_len: int
):
    """
    Combines and preprocesses corpus.
    
    Args:
        input_dir: Directory containing source .jsonl files.
        output_file: Path to save the combined .jsonl file.
        min_length: Minimum character length of a document to keep.
        dedup_mode: 'document' (exact match) or 'paragraph' (dedup paragraphs).
        sequence_min_len: Minimum length for a paragraph/sequence to be considered for deduplication.
    """
    
    seen_doc_hashes: Set[str] = set()
    seen_paragraph_hashes: Set[str] = set()
    
    total_docs = 0
    kept_docs = 0
    total_chars_in = 0
    total_chars_out = 0
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for doc in load_jsonl_files(input_dir):
            total_docs += 1
            text = doc.get("text", "")
            
            if not isinstance(text, str) or len(text) < min_length:
                continue
                
            total_chars_in += len(text)
            
            # 1. Document-level Exact Deduplication
            doc_hash = get_content_hash(text)
            if doc_hash in seen_doc_hashes:
                continue
            seen_doc_hashes.add(doc_hash)
            
            # 2. Content Processing
            if dedup_mode == 'paragraph':
                # Split by newlines to identify paragraphs/lines
                lines = text.split('\n')
                unique_lines = []
                
                for line in lines:
                    line_stripped = line.strip()
                    if len(line_stripped) < sequence_min_len:
                        # Keep short lines (likely sentence fragments or dialogue), 
                        # don't try to dedup them globally as they might be common phrases.
                        unique_lines.append(line)
                        continue
                        
                    line_hash = get_content_hash(line_stripped)
                    if line_hash not in seen_paragraph_hashes:
                        seen_paragraph_hashes.add(line_hash)
                        unique_lines.append(line)
                
                if not unique_lines:
                    continue
                    
                text = "\n".join(unique_lines)
                
                # Re-check length after paragraph dedup
                if len(text) < min_length:
                    continue
            
            # Update doc with processed text
            doc["text"] = text
            
            # Write to output
            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept_docs += 1
            total_chars_out += len(text)

    logger.info("-" * 40)
    logger.info(f"Processing Complete.")
    logger.info(f"Input Documents: {total_docs}")
    logger.info(f"Output Documents: {kept_docs}")
    logger.info(f"Reduction in Docs: {100 * (1 - kept_docs/max(1, total_docs)):.2f}%")
    logger.info(f"Input Characters: {total_chars_in}")
    logger.info(f"Output Characters: {total_chars_out}")
    logger.info(f"Reduction in Chars: {100 * (1 - total_chars_out/max(1, total_chars_in)):.2f}%")
    logger.info(f"Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Combine and preprocess JSONL corpus files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .jsonl files")
    parser.add_argument("--output-file", type=Path, required=True, help="Output .jsonl file path")
    parser.add_argument("--min-length", type=int, default=100, help="Minimum text length to keep a document")
    parser.add_argument("--dedup-mode", choices=['document', 'paragraph'], default='paragraph', 
                        help="Deduplication mode. 'document' for exact doc match, 'paragraph' to remove repeated paragraphs across corpus.")
    parser.add_argument("--sequence-min-len", type=int, default=50, 
                        help="Minimum length of a sequence/paragraph to be considered for global deduplication (only used in paragraph mode).")
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        logger.error(f"Input directory {args.input_dir} does not exist.")
        sys.exit(1)
        
    preprocess_and_combine(
        args.input_dir,
        args.output_file,
        args.min_length,
        args.dedup_mode,
        args.sequence_min_len
    )

if __name__ == "__main__":
    main()
