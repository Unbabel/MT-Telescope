# MT-Telescope

MT Telescope is an improved version of ml-metrics where you can evaluate your MT systems and perform pairwise comparisons between them with your command line and/or a web browser.

## Install:
Create a virtual environment and make sure you have [poetry](https://python-poetry.org/docs/#installation) installed.

Finally run:

```bash
git clone https://github.com/Unbabel/MT-Telescope
cd MT-Telescope
poetry install
```

## Scoring:

To get the system level scores for a particular MT simply run `telescope score`.

```bash

```

## Comparing two systems:
For comparison between two systems you can run telescope using:
1. The command line interface
2. A web browser

### Command Line Interface (CLI):

For running system comparisons with CLI you should use the `telescope compare` command.

```
Usage: telescope compare [OPTIONS]

Options:
  -s, --source FILENAME           Source segments.  [required]
  -x, --system_x FILENAME         System X MT outputs.  [required]
  -y, --system_y FILENAME         System Y MT outputs.  [required]
  -r, --reference FILENAME        Reference segments.  [required]
  -l, --language TEXT             Language of the evaluated text.  [required]
  -m, --metric [COMET|sacreBLEU|chrF|ZeroEdit|BLEURT|BERTScore|TER|Prism|BLEU|GLEU]
                                  MT metric to run.  [required]
  -f, --filter [named-entities|duplicates]
                                  MT metric to run.
  --seg_metric [COMET|ZeroEdit|BLEURT|BERTScore|Prism|GLEU]
                                  Segment-level metric to use for segment-
                                  level analysis.

  -o, --output_folder TEXT        Folder you wish to use to save plots.
  --bootstrap
  --num_splits INTEGER            Number of random partitions used in
                                  Bootstrap resampling.

  --sample_ratio FLOAT            Folder you wish to use to save plots.
  --help                          Show this message and exit.
```

#### Example 1: Running several metrics

Running BLEU, chrF BERTScore and COMET to compare two systems:

```bash
telescope compare \
  -s path/to/src/file.txt \
  -x path/to/system-x/file.txt \
  -y path/to/system-y \
  -r path/to/ref/file.txt \
  -l en \
  -m BLEU -m chrF -m BERTScore
```

#### Example 2: Saving a comparison report

```bash
telescope compare \
  -s path/to/src/file.txt \
  -x path/to/system-x/file.txt \
  -y path/to/system-y \
  -r path/to/ref/file.txt \
  -l en \
  -m COMET \
  --output_folder FOLDER-PATH
```

### Web Interface

To run a web interface simply run:
```bash
streamlit run app.py
```
