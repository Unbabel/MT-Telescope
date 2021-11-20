<p align="center">
  <img src="https://user-images.githubusercontent.com/17256847/124762084-66212200-df2a-11eb-92ce-edbebfe9d4e2.jpg">
  <br />
  <br />
  <a href="https://github.com/Unbabel/MT-Telescope/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Unbabel/MT-Telescope" /></a>
  <a href="https://github.com/Unbabel/MT-Telescope/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Unbabel/MT-Telescope" /></a>
  <a href=""><img alt="PyPI" src="https://img.shields.io/pypi/v/mt-telescope" /></a>
  <a href="https://github.com/psf/black"><img alt="Code Style" src="https://img.shields.io/badge/code%20style-black-black" /></a>
</p>

# MT-Telescope

MT-Telescope is a toolkit for comparative analysis of MT systems that provides a number of tools that add rigor and depth to MT evaluation. With this package we endeavour to make it easier for researchers and industry practitioners to compare MT systems by giving you easy access to:

1) SOTA MT evaluation metrics such as COMET  [(rei, et al 2020)](https://aclanthology.org/2020.emnlp-main.213/).
2) Statistical tests such as bootstrap resampling [(Koehn, et al 2004)](https://aclanthology.org/W04-3250/).
3) Dynamic Filters to select parts of your testset with specific phenomena
4) Visual interface/plots to compare systems side-by-side segment-by-segment.

We highly recommend reading the following papers to learn more about how to perform better MT-Evaluation:
- [Scientific Credibility of Machine Translation Research: A Meta-Evaluation of 769 Papers](https://arxiv.org/pdf/2106.15195.pdf)
- [To Ship or Not to Ship: An Extensive Evaluation of Automatic Metrics for Machine Translation](https://arxiv.org/pdf/2107.10821.pdf)


## Install:

### Via pip:

```bash
pip install mt-telescope
```

Note: This is a pre-release currently.

### Locally:
Create a virtual environment and make sure you have [poetry](https://python-poetry.org/docs/#installation) installed.

Finally run:

```bash
git clone https://github.com/Unbabel/MT-Telescope
cd MT-Telescope
poetry install --no-dev
```

## Scoring:

To get the system level scores for a particular MT simply run `telescope score`.

```bash
telescope score -s {path/to/sources} -t {path/to/translations} -r {path/to/references} -l {target_language} -m COMET -m chrF
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
  -m, --metric [COMET|sacreBLEU|chrF|ZeroEdit|BERTScore|TER|Prism|GLEU]
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
  -m BLEU -m chrF -m BERTScore -m COMET
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
telescope streamlit
```

Some metrics like COMET can take some time to run inside streamlit. You can switch the COMET model to a more lightweight model with the following env variable:
```bash
export COMET_MODEL=wmt21-cometinho-da
```

# Cite:

```
@inproceedings{rei-etal-2021-mt,
    title = "{MT}-{T}elescope: {A}n interactive platform for contrastive evaluation of {MT} systems",
    author = {Rei, Ricardo  and  Stewart, Craig  and  Farinha, Ana C  and  Lavie, Alon},
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-demo.9",
    doi = "10.18653/v1/2021.acl-demo.9",
    pages = "73--80",
}
```
