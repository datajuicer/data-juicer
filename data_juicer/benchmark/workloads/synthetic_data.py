#!/usr/bin/env python3
"""
Synthetic data generation for benchmark workloads.

Generates test data when production datasets are not available,
useful for CI/CD pipelines and quick testing.
"""

import json
import random
import string
from pathlib import Path

from loguru import logger


class SyntheticDataGenerator:
    """Generator for synthetic benchmark data."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def generate_text_data(
        self,
        output_path: str,
        num_samples: int = 1000,
        min_length: int = 10,
        max_length: int = 3000,
    ) -> str:
        """Generate synthetic text data for benchmarking.

        Creates text samples with varying characteristics to test filters:
        - Various text lengths (some will be filtered by text_length_filter)
        - Various word counts (some will be filtered by words_num_filter)
        - Some with special characters (will be affected by special_characters_filter)
        - Some with repeated characters (will be affected by character_repetition_filter)

        Args:
            output_path: Path to write the generated data
            num_samples: Number of samples to generate
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters

        Returns:
            Path to the generated file
        """
        random.seed(self.seed)

        samples = []
        for i in range(num_samples):
            # Vary text length
            text_len = random.randint(min_length, max_length)

            # Generate base text
            words = []
            current_len = 0
            while current_len < text_len:
                word_len = random.randint(2, 12)
                word = "".join(random.choices(string.ascii_lowercase, k=word_len))
                words.append(word)
                current_len += word_len + 1

            text = " ".join(words)[:text_len]

            # Add some structure (30% of samples)
            if random.random() < 0.3:
                text = f"Title: Sample {i}\n\n{text}\n\nConclusion: End of sample."

            # Add special characters to some samples (20%)
            if random.random() < 0.2:
                special_chars = "!@#$%^&*()[]{}|;:',.<>?/"
                insert_pos = random.randint(0, len(text) - 1)
                num_special = random.randint(5, 20)
                special = "".join(random.choices(special_chars, k=num_special))
                text = text[:insert_pos] + special + text[insert_pos:]

            # Add repeated characters to some samples (15%)
            if random.random() < 0.15:
                repeat_char = random.choice(string.ascii_lowercase)
                repeat_len = random.randint(15, 30)
                insert_pos = random.randint(0, len(text) - 1)
                text = text[:insert_pos] + (repeat_char * repeat_len) + text[insert_pos:]

            sample = {
                "text": text,
                "meta": {
                    "id": i,
                    "source": random.choice(["wiki", "arxiv", "web", "book"]),
                    "synthetic": True,
                },
            }
            samples.append(json.dumps(sample, ensure_ascii=False))

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(samples) + "\n")

        logger.info(f"Generated {num_samples} synthetic samples to {output_path}")
        return str(output_path)

    def generate_or_get_data(
        self,
        output_path: str,
        num_samples: int = 1000,
        force_regenerate: bool = False,
    ) -> str:
        """Generate data if it doesn't exist, or return existing path.

        Args:
            output_path: Path to write/check the data
            num_samples: Number of samples to generate if needed
            force_regenerate: Force regeneration even if file exists

        Returns:
            Path to the data file
        """
        output_path = Path(output_path)

        if output_path.exists() and not force_regenerate:
            # Check if file has expected number of samples
            with open(output_path, "r") as f:
                existing_samples = sum(1 for _ in f)
            if existing_samples >= num_samples:
                logger.info(f"Using existing synthetic data: {output_path} ({existing_samples} samples)")
                return str(output_path)

        return self.generate_text_data(str(output_path), num_samples)


# Global instance for convenience
SYNTHETIC_DATA_GENERATOR = SyntheticDataGenerator()
