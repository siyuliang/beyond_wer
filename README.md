# Whisper Beam Search Analysis

This repository contains code for analyzing OpenAI's Whisper large-v2 model performance across multiple languages using beam search decoding with detailed token-level analysis.

## Overview

The analysis examines Whisper's behavior during beam search decoding by capturing:
- Token-level predictions and probabilities
- Confidence scores of chosen tokens
- Entropy distributions over candidate tokens
- Diversity of alternate hypotheses
- Gold token rankings (via alignment with ground truth)
- Word Error Rate (WER) vs training hours

## Data Collection

**Input**: Common Voice 17.0 dataset (~600 seconds per language)  
**Model**: Whisper large-v2  
**Decoding**: Beam search (beam_size=5, temperature=0.2)  
**Languages**: 30 languages across high/medium/low resource groups

### Language Coverage

- **High-resource** (5): German, Spanish, French, Portuguese, Turkish
- **Medium-resource** (13): Italian, Dutch, Swedish, Catalan, Finnish, Indonesian, Vietnamese, Romanian, Norwegian, Czech, Hungarian, Yoruba
- **Low-resource** (12): Welsh, Lithuanian, Latvian, Azerbaijani, Estonian, Basque

**Excluded languages**: Uzbek, Maltese, Swahili, Albanian, Yoruba, Danish, Vietnamese, Czech

## Repository Structure

```
emnlp/
├── README.md                           # This file
├── subtoken_beam.ipynb                 # Main data collection script
├── plot_confidence.ipynb               # Confidence vs training hours analysis
├── plot_diversity.ipynb                # Type-Token Ratio of alternates
├── plot_entropy.ipynb                  # Token entropy analysis
├── plot_rank.ipynb                     # Gold token rank analysis
├── plot_wer.ipynb                      # WER vs training hours
├── whisper_training_hours.csv          # Training data metadata
├── results_beam_600s/                  # Raw beam search data (CSV per language)
└── analysis_results_beam/              # Generated plots and metrics
```

## Workflow

### 1. Data Generation (`subtoken_beam.ipynb`)

Generates token-level beam search dumps for each language:
- Captures top K=50 candidate tokens per decoding step
- Records probabilities, chosen tokens, and full transcriptions
- Outputs: `results_beam_600s/*_subtoken_beam.csv`

**Key columns**:
- `step`: Decoding step (-1 = final transcription)
- `top_k_subtokens`: Top 50 candidate tokens (JSON array)
- `top_k_probs`: Corresponding probabilities
- `chosen_subtoken`: Token selected by beam search
- `chosen_rank`: Rank of chosen token (1-50)
- `ground_truth`: Reference transcription
- `full_transcription`: Hypothesis transcription
- `whisper_lang`: Whisper language code

### 2. Analysis Scripts

Each notebook reads from `results_beam_600s/` and generates visualizations in `analysis_results_beam/`:

#### **`plot_confidence.ipynb`**
- Calculates average confidence (probability) of chosen tokens
- Plots confidence vs Whisper training hours
- Outputs: Scatter plot with Pearson correlation

#### **`plot_diversity.ipynb`**
- Computes Type-Token Ratio (TTR) of alternate candidates (top2-topK)
- Measures lexical diversity of beam search alternates
- Outputs: Diversity metrics and visualization

#### **`plot_entropy.ipynb`**
- Calculates average Shannon entropy over top K_H=50 probabilities
- Analyzes prediction uncertainty
- Outputs: Entropy distribution plots

#### **`plot_rank.ipynb`**
- Aligns predictions with ground truth using Levenshtein algorithm
- Finds rank of gold tokens in beam search candidates
- Outputs: Gold token rank statistics and plots

#### **`plot_wer.ipynb`**
- Computes Word Error Rate per language
- Generates multiple plot types:
  - Scatter: WER vs training hours (log scale) with trend line
  - Ranked bar: Languages sorted by WER
  - Box plot: WER by resource group
- Outputs: `language_wer_metrics.csv` + PNG/PDF/SVG plots

## Usage

### Prerequisites
```bash
pip install numpy pandas matplotlib scipy jiwer Levenshtein transformers datasets
```

### Running the Analysis

1. **Generate beam search data** (if not already done):
   ```python
   # Run subtoken_beam.ipynb
   # Outputs: results_beam_600s/*_subtoken_beam.csv
   ```

2. **Run individual analyses**:
   ```python
   # Run any plot_*.ipynb notebook
   # Each notebook is self-contained and processes all languages
   ```

3. **View results**:
   - Check `analysis_results_beam/` for generated plots
   - Review `language_wer_metrics.csv` for aggregate statistics

## Configuration

All notebooks share consistent configuration:

```python
# Data parameters
K_D = 50           # Diversity top-K candidates
K_H = 50           # Entropy/hypothesis top-K
TARGET_SEC = 600   # Target seconds per language
BEAM_SIZE = 5      # Beam search width
TEMPERATURE = 0.2  # Sampling temperature

# Paths
DATA_DIR = Path("results_beam_600s")
OUTPUT_DIR = Path("analysis_results_beam")
TRAINING_HOURS_CSV = Path("whisper_training_hours.csv")
```

## Output Files

### `results_beam_600s/`
- `*_subtoken_beam.csv`: Per-language beam search dumps (28 files)

### `analysis_results_beam/`
- `confidence_vs_hours.png/pdf/svg`: Confidence analysis
- `diversity_metrics.csv`: TTR statistics
- `entropy_distribution.png`: Entropy plots
- `gold_rank_statistics.csv`: Rank analysis
- `wer_vs_hours.png/pdf/svg`: WER scatter plot
- `languages_ranked_by_wer.png/pdf`: Ranked bar chart
- `wer_by_resource_group.png/pdf`: Box plot
- `language_wer_metrics.csv`: Aggregate WER metrics

## Notes
Part of this text is generated with LLM