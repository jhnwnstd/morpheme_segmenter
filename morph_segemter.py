"""
Morphological Analysis System
----------------------------
A tool for analyzing morphological patterns in text using statistical methods.
"""

import nltk
from nltk.corpus import brown, wordnet
from collections import defaultdict
import math
import numpy as np
from typing import Dict, List, Set, DefaultDict
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.stats import chi2_contingency
import random
import logging
from tqdm import tqdm
import argparse
import sys
import pandas as pd

# Type aliases for clarity
Matrix = DefaultDict[str, DefaultDict[str, Dict[str, str]]]
ScoreDict = Dict[str, float]


@dataclass
class AnalysisParams:
    """Parameters for morphological analysis."""
    subset_size: int = 100000
    max_affix_length: int = 5
    min_affix_length: int = 2
    max_stem_length: int = 10
    min_stem_length: int = 3
    min_frequency: int = 10  # Frequency threshold for affixes
    min_occurrences: int = 3
    is_percentile: float = 60.0
    chi_square_percentile: float = 60.0
    num_samples: int = 10
    log_level: str = "INFO"
    export_path: str = "morph_matrix.csv"


class MorphologicalAnalyzer:
    """Main class for morphological analysis."""

    def __init__(self, params: AnalysisParams):
        """Initialize the analyzer with given parameters."""
        self.params = params
        self.unique_words: Set[str] = set()
        self.corpus_subset: List[str] = []
        self.candidate_affixes: Set[str] = set()
        self.M: Matrix = defaultdict(lambda: defaultdict(dict))
        self.IS: ScoreDict = {}
        self.chi_square_scores: ScoreDict = {}
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up the logger for the analyzer."""
        logger = logging.getLogger("MorphologicalAnalyzer")
        logger.setLevel(getattr(logging, self.params.log_level.upper(), logging.INFO))
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        return logger

    def setup_corpus(self) -> None:
        """Initialize and preprocess the corpus."""
        self.logger.info("Downloading required NLTK data...")
        nltk.download('brown', quiet=True)
        nltk.download('wordnet', quiet=True)

        self.logger.info("Loading and preprocessing the Brown corpus...")
        corpus_words = brown.words()
        processed_words = [word.lower() for word in corpus_words if word.isalpha()]
        self.corpus_subset = processed_words[:self.params.subset_size]
        self.unique_words = set(self.corpus_subset)

        self.logger.info(f"Corpus loaded: {len(self.corpus_subset)} words")
        self.logger.info(f"Unique words: {len(self.unique_words)}\n")

    def identify_affixes(self) -> None:
        """Identify candidate affixes from the corpus with enhanced frequency filtering."""
        self.logger.info("Identifying candidate affixes...")
        prefix_freq = defaultdict(int)
        suffix_freq = defaultdict(int)

        for word in tqdm(self.unique_words, desc="Processing words for affixes", unit="words"):
            word_len = len(word)
            for l in range(self.params.min_affix_length, min(self.params.max_affix_length + 1, word_len)):
                prefix = word[:l]
                suffix = word[-l:]
                prefix_freq[prefix] += 1
                suffix_freq[suffix] += 1

        min_freq = self.params.min_frequency
        self.candidate_affixes = {
            affix
            for affix, freq in {**prefix_freq, **suffix_freq}.items()
            if freq >= min_freq
        }

        # Log details for debugging
        self.logger.info(f"Identified {len(self.candidate_affixes)} candidate affixes with a min frequency of {min_freq}\n")

    def build_matrix(self) -> None:
        """Build the morphological matrix M with enhanced filtering."""
        self.logger.info("Building the morphological matrix...")
        for stem in tqdm(self.unique_words, desc="Building matrix", unit="stems"):
            stem_len = len(stem)

            # Skip stems that do not meet length constraints
            if stem_len < self.params.min_stem_length or stem_len > self.params.max_stem_length:
                continue

            for affix in self.candidate_affixes:
                affix_len = len(affix)
                if affix_len >= stem_len:
                    continue

                # Check for suffix
                if stem.endswith(affix):
                    base = stem[:-affix_len]
                    if base in self.unique_words and self.params.min_stem_length <= len(base) <= self.params.max_stem_length:
                        self.M[base][stem] = {'affix': affix, 'type': 'suffix'}

                # Check for prefix
                if stem.startswith(affix):
                    base = stem[affix_len:]
                    if base in self.unique_words and self.params.min_stem_length <= len(base) <= self.params.max_stem_length:
                        self.M[base][stem] = {'affix': affix, 'type': 'prefix'}

        # Filter stems by length
        self.M = defaultdict(lambda: defaultdict(dict), {
            stem: words for stem, words in self.M.items()
            if self.params.min_stem_length <= len(stem) <= self.params.max_stem_length
        })

        self.logger.info(f"Matrix built with {len(self.M)} stems\n")

    def calculate_scores(self) -> None:
        """Calculate all morphological scores (IS, Chi-Square)."""
        self.logger.info("Calculating Independent Scores (IS)...")
        self._calculate_IS()
        self.logger.info("Calculating Chi-Square scores...")
        self._calculate_chi_square()
        self.logger.info("All scores calculated.\n")

    def _calculate_IS(self) -> None:
        """Calculate Independent Scores."""
        affix_stems = defaultdict(set)
        affix_freq = defaultdict(int)

        for stem, words in self.M.items():
            for word, info in words.items():
                affix = info['affix']
                affix_stems[affix].add(stem)
                affix_freq[affix] += 1

        self.IS = {
            affix: math.tanh(len(stems)) * math.log1p(affix_freq[affix])
            for affix, stems in affix_stems.items()
        }

        # Print sample scores
        sample_affixes = list(self.IS.keys())[:self.params.num_samples]
        self.logger.info("Sample Independent Scores:")
        for affix in sample_affixes:
            self.logger.info(f"Affix: '{affix}' - IS: {self.IS[affix]:.4f}")
        self.logger.info("")

    def _calculate_chi_square(self) -> None:
        """Calculate Chi-Square statistics for affixes."""
        self.logger.debug("Calculating Chi-Square scores...")
        total_attachments = sum(len(words) for words in self.M.values())
        total_stems = len(self.M)

        affix_counts = defaultdict(int)
        for stem, words in self.M.items():
            for word, info in words.items():
                affix = info['affix']
                affix_counts[affix] += 1

        self.chi_square_scores = {}
        for affix in self.candidate_affixes:
            a = affix_counts[affix]  # Affix present
            b = total_attachments - a  # Affix absent
            c = len([stem for stem in self.M
                     if affix in [info['affix'] for info in self.M[stem].values()]])  # Stems with affix
            d = total_stems - c  # Stems without affix

            contingency_table = [[c, d], [a, b]]
            try:
                chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
                self.chi_square_scores[affix] = chi2
            except ValueError as e:
                self.logger.warning(f"Chi-Square calculation failed for affix '{affix}': {e}")
                self.chi_square_scores[affix] = 0.0

    def handle_multi_slot_affixes(self) -> None:
        """Handle multi-slot morphological cases."""
        self.logger.info("Handling multi-slot affixes (affixes with IS=0)...")
        zero_IS_affixes = [affix for affix, score in self.IS.items() if score == 0.0]
        self.logger.info(f"Number of affixes with IS=0: {len(zero_IS_affixes)}\n")

        for affix in zero_IS_affixes:
            segments = self._get_all_segmentations(affix)
            if segments:
                mslot_IS_scores = []
                for seg in segments:
                    # Only consider segmentations where all parts have IS > 0
                    valid = all(self.IS.get(s, 0.0) > 0.0 for s in seg)
                    if not valid:
                        continue
                    mslot_IS = sum(self.IS[axi] for axi in seg) / len(seg)
                    mslot_IS_scores.append((mslot_IS, seg))

                if mslot_IS_scores:
                    best_mslot_IS, best_segmentation = max(mslot_IS_scores, key=lambda x: x[0])
                    self.IS[affix] = best_mslot_IS
                    self.logger.info(f"Affix '{affix}' decomposed into {best_segmentation} "
                                     f"with mslot_IS={best_mslot_IS:.4f}")
            else:
                self.logger.debug(f"No valid segmentations found for affix '{affix}'")
        self.logger.info("Multi-slot affix handling completed.\n")

    def _get_all_segmentations(self, affix: str) -> List[List[str]]:
        """Get all possible segmentations of an affix."""
        n = len(affix)
        segmentations = []

        # Generate all possible non-trivial segmentations
        for num_splits in range(1, min(3, n)):  # Limit to 2 splits to reduce complexity
            for positions in self._get_split_positions(n, num_splits):
                splits = []
                last_pos = 0
                for pos in positions:
                    splits.append(affix[last_pos:pos])
                    last_pos = pos
                splits.append(affix[last_pos:])

                if all(self.IS.get(s, 0.0) > 0.0 for s in splits):
                    segmentations.append(splits)

        return segmentations

    @staticmethod
    def _get_split_positions(n: int, k: int) -> List[List[int]]:
        """Generate all possible split positions."""
        from itertools import combinations
        return list(combinations(range(1, n), k))

    def generate_corpus_absent_stems(self) -> None:
        """Generate and validate potential stems not present in corpus."""
        self.logger.info("Generating and validating potential stems not present in the corpus...")
        wordnet_words = set(wordnet.words())
        non_zero_IS_affixes = [affix for affix, score in self.IS.items() if score > 0.0]
        new_stems_added = 0

        for affix in tqdm(non_zero_IS_affixes, desc="Generating stems", unit="affixes"):
            affix_length = len(affix)
            stem_counts = defaultdict(int)

            # Count potential stems
            for word in self.unique_words:
                word_length = len(word)
                if word_length <= affix_length:
                    continue

                # Check suffix case
                if word.endswith(affix):
                    new_stem = word[:-affix_length]
                    stem_counts[new_stem] += 1

                # Check prefix case
                if word.startswith(affix):
                    new_stem = word[affix_length:]
                    stem_counts[new_stem] += 1

            # Add valid stems
            for new_stem, count in stem_counts.items():
                if (count >= self.params.min_occurrences and
                        new_stem not in self.unique_words and
                        new_stem in wordnet_words):
                    # Update matrix for suffixes
                    for word in self.unique_words:
                        if word.endswith(affix) and word.startswith(new_stem):
                            self.M[new_stem][word] = {'affix': affix, 'type': 'suffix'}

                        if word.startswith(affix) and word.endswith(new_stem):
                            self.M[new_stem][word] = {'affix': affix, 'type': 'prefix'}

                    new_stems_added += 1

        self.logger.info(f"Added {new_stems_added} new stems after validation\n")

    def validate_stems(self) -> None:
        """Validate stems against WordNet."""
        self.logger.info("Validating new stems against WordNet...")
        new_stems = [stem for stem in self.M if stem not in self.corpus_subset]
        num_checks = min(self.params.num_samples, len(new_stems))

        if num_checks == 0:
            self.logger.info("No new stems to validate.\n")
            return

        self.logger.info(f"Validating {num_checks} new stems:")
        for stem in random.sample(new_stems, num_checks):
            is_valid = bool(wordnet.synsets(stem))
            validity = "Valid" if is_valid else "Invalid"
            self.logger.info(f"Stem: '{stem}' - {validity}")
        self.logger.info("Stem validation completed.\n")

    def filter_affixes(self) -> None:
        """Filter affixes based on IS and Chi-Square scores."""
        self.logger.info("Filtering affixes based on IS and Chi-Square thresholds...")
        is_threshold = self._get_dynamic_threshold(
            list(self.IS.values()),
            self.params.is_percentile
        )
        chi_square_threshold = self._get_dynamic_threshold(
            list(self.chi_square_scores.values()),
            self.params.chi_square_percentile
        )

        self.logger.info(f"IS threshold (percentile {self.params.is_percentile}): {is_threshold:.4f}")
        self.logger.info(f"Chi-Square threshold (percentile {self.params.chi_square_percentile}): {chi_square_threshold:.4f}\n")

        # Filter affixes
        significant_is = {
            affix for affix, score in self.IS.items()
            if score >= is_threshold
        }
        significant_chi_square = {
            affix for affix, score in self.chi_square_scores.items()
            if score >= chi_square_threshold
        }

        # Keep only affixes that pass both filters
        significant_affixes = significant_is.intersection(significant_chi_square)

        self.logger.info(f"Affixes after filtering: {len(significant_affixes)} out of {len(self.candidate_affixes)}")

        # Update matrix by removing words with non-significant affixes
        M_filtered = defaultdict(lambda: defaultdict(dict))
        for stem, words in self.M.items():
            for word, info in words.items():
                if info['affix'] in significant_affixes:
                    M_filtered[stem][word] = info

        self.M = M_filtered
        self.logger.info(f"Matrix filtered to {len(self.M)} stems\n")

    @staticmethod
    def _get_dynamic_threshold(scores: List[float], percentile: float) -> float:
        """Calculate dynamic threshold based on score distribution."""
        if not scores:
            return 0.0
        return max(np.percentile(scores, percentile), 1.0)

    def print_sample_matrix(self) -> None:
        """Print sample entries from the morphological matrix."""
        self.logger.info("Sample entries from Matrix M:")
        sample_stems = list(self.M.keys())[:self.params.num_samples]

        for stem in sample_stems:
            self.logger.info(f"\nStem: '{stem}'")
            for word, info in self.M[stem].items():
                self.logger.info(f"  Word: '{word}' - Affix: '{info['affix']}' ({info['type']})")
        self.logger.info("")

    def export_matrix_to_csv(self) -> None:
        """Export the morphological matrix to a CSV file."""
        self.logger.info(f"Exporting morphological matrix to '{self.params.export_path}'...")
        rows = []
        for stem, words in self.M.items():
            for word, info in words.items():
                rows.append({
                    'Stem': stem,
                    'Word': word,
                    'Affix': info['affix'],
                    'Type': info['type']
                })

        df = pd.DataFrame(rows)
        try:
            df.to_csv(self.params.export_path, index=False)
            self.logger.info("Morphological matrix exported successfully.\n")
        except Exception as e:
            self.logger.error(f"Failed to export matrix: {e}\n")

    def visualize(self) -> None:
        """Generate all visualizations."""
        self.logger.info("Generating visualizations...")
        self._plot_score_distributions()
        self._plot_top_affixes()
        self._plot_frequencies()
        self.logger.info("Visualizations generated.\n")

    def _plot_score_distributions(self) -> None:
        """Plot distributions of all scores."""
        self.logger.info("Plotting score distributions...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot IS distribution
        axes[0].hist(list(self.IS.values()), bins=50, color='skyblue', edgecolor='black')
        axes[0].set_title('Distribution of Independent Scores (IS)')
        axes[0].set_xlabel('Score')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True)

        # Plot Chi-Square distribution
        axes[1].hist(list(self.chi_square_scores.values()), bins=50, color='violet', edgecolor='black')
        axes[1].set_title('Distribution of Chi-Square Scores')
        axes[1].set_xlabel('Score')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def _plot_top_affixes(self, top_n: int = 20) -> None:
        """Plot top N affixes by different metrics."""
        self.logger.info("Plotting top affixes by different metrics...")
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Plot top IS affixes
        self._plot_top_metric(self.IS, "Independent Score", axes[0], top_n)

        # Plot top Chi-Square affixes
        self._plot_top_metric(self.chi_square_scores, "Chi-Square", axes[1], top_n)

        plt.tight_layout()
        plt.show()

    def _plot_top_metric(self, metric_dict: Dict[str, float],
                         title: str, ax: plt.Axes, top_n: int) -> None:
        """Helper function for plotting top affixes by a given metric."""
        sorted_items = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        if not sorted_items:
            self.logger.warning(f"No data to plot for {title}.")
            return

        affixes, scores = zip(*sorted_items)
        ax.bar(range(len(affixes)), scores, color='skyblue', edgecolor='black')
        ax.set_title(f'Top {top_n} Affixes by {title}')
        ax.set_xlabel('Affix')
        ax.set_ylabel('Score')
        ax.set_xticks(range(len(affixes)))
        ax.set_xticklabels(affixes, rotation=45, ha='right')
        ax.grid(True)

    def _plot_frequencies(self) -> None:
        """Plot affix frequencies."""
        self.logger.info("Plotting affix frequencies...")
        affix_frequencies = defaultdict(int)
        for stem, words in self.M.items():
            for word, info in words.items():
                affix = info['affix']
                affix_frequencies[affix] += 1

        sorted_affixes = sorted(affix_frequencies.items(),
                                key=lambda x: x[1], reverse=True)[:20]

        if sorted_affixes:
            affixes, frequencies = zip(*sorted_affixes)

            plt.figure(figsize=(14, 8))
            plt.bar(range(len(affixes)), frequencies, color='lightgreen', edgecolor='black')
            plt.title('Top 20 Affixes by Frequency')
            plt.xlabel('Affix')
            plt.ylabel('Frequency')
            plt.xticks(range(len(affixes)), affixes, rotation=45, ha='right')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            self.logger.warning("No affix frequencies to plot.")

    def run_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        self.setup_corpus()
        self.identify_affixes()
        self.build_matrix()
        self.calculate_scores()
        self.handle_multi_slot_affixes()
        self.generate_corpus_absent_stems()
        self.filter_affixes()
        self.print_sample_matrix()
        self.validate_stems()
        self.export_matrix_to_csv()
        self.visualize()

def parse_arguments() -> AnalysisParams:
    """Parse command-line arguments to set analysis parameters."""
    parser = argparse.ArgumentParser(description="Morphological Analysis System")
    parser.add_argument('--subset_size', type=int, default=100000,
                        help='Size of the corpus subset (default: 100000)')
    parser.add_argument('--max_affix_length', type=int, default=5,
                        help='Maximum length of affixes to consider (default: 5)')
    parser.add_argument('--min_affix_length', type=int, default=2,
                        help='Minimum length of affixes to consider (default: 2)')
    parser.add_argument('--min_frequency', type=int, default=5,
                        help='Minimum frequency of affixes to be considered (default: 5)')
    parser.add_argument('--min_occurrences', type=int, default=3,
                        help='Minimum occurrences for stems not present in corpus (default: 3)')
    parser.add_argument('--is_percentile', type=float, default=60.0,
                        help='Percentile threshold for Independent Scores (default: 60.0)')
    parser.add_argument('--chi_square_percentile', type=float, default=60.0,
                        help='Percentile threshold for Chi-Square scores (default: 60.0)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to display for matrix and validation (default: 10)')
    parser.add_argument('--log_level', type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help='Logging level (default: INFO)')
    parser.add_argument('--export_path', type=str, default="morph_matrix.csv",
                        help='Path to export the morphological matrix CSV (default: morph_matrix.csv)')
    args = parser.parse_args()

    return AnalysisParams(
        subset_size=args.subset_size,
        max_affix_length=args.max_affix_length,
        min_affix_length=args.min_affix_length,
        min_frequency=args.min_frequency,
        min_occurrences=args.min_occurrences,
        is_percentile=args.is_percentile,
        chi_square_percentile=args.chi_square_percentile,
        num_samples=args.num_samples,
        log_level=args.log_level,
        export_path=args.export_path
    )


def main():
    """Main execution function."""
    params = parse_arguments()

    # Create analyzer
    analyzer = MorphologicalAnalyzer(params)

    # Run analysis pipeline
    analyzer.run_analysis()


if __name__ == "__main__":
    main()