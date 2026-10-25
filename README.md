# Review of AI/ML research for fingerprint deduplication

**Sreejith Markkassery · October 2026**

Review of three papers for billion-scale fingerprint deduplication, with a partial-print experiment:

1. **AFR-Net**: Attention-Driven Fingerprint Recognition Network. Grosz & Jain, IEEE T-BIOM 2024 ([arXiv 2211.13897](https://arxiv.org/abs/2211.13897))
2. **JIPNet**: Joint Identity Verification and Pose Alignment for Partial Fingerprints. Guan, Pan, Feng, Zhou, IEEE TIFS 2025 ([arXiv 2405.03959](https://arxiv.org/abs/2405.03959))
3. **HERS**: Homomorphically Encrypted Representation Search. Engelsma, Jain, Boddeti, IEEE T-BIOM 2022 ([arXiv 2003.12197](https://arxiv.org/abs/2003.12197))

**Tasks addressed**
- **(a)** Applicability of each paper to billion-scale gallery development: design choices, latency (wall-clock and machine-independent complexity), failure cases; a proposed staged pipeline combining the three.
- **(b)** JIPNet and an AFR-Net-style baseline on FVC2002/2004 partial prints: EER, TAR at FAR 0.1 %, JIPNet pose error, failure analysis.

## Report

| File | Description |
|---|---|
| [`reports/UIDAI_review_submission.pdf`](reports/UIDAI_review_submission.pdf) | Final report (A4, 10 pages) |
| [`reports/latex/UIDAI_review.tex`](reports/latex/UIDAI_review.tex) | LaTeX source of the report (figures in `reports/latex/figures/`) |
| [`reports/UIDAI_review_submission.html`](reports/UIDAI_review_submission.html) | Same report as a web page |

## Key results (part b)

| | JIPNet | AFR-Net-style baseline |
|---|---|---|
| EER, six real-sensor databases | 1.6 – 3.2 % | — |
| TAR at FAR 0.1 %, real sensors | 81 – 88 % | — |
| EER, FVC2002 DB1 / DB3 (same 400 genuine pairs) | 2.2 % / 3.0 % | 36.5 % / 37.5 % |
| Typical pose error | 7 – 11° rotation, 8 – 13 px (0.4 – 0.7 mm) translation | — |

Both models agree with the figures reported in the JIPNet paper (Table V). Full tables, plots and failure analysis are in the report.

## Repository structure

```
src/                     code (see pipeline below)
data/partial/            test-pair definitions per database (image, patch centre, rotation, ground-truth pose)
experiments/runs/        JIPNet scores and poses for every pair (8 databases × 3,200 pairs)
experiments/runs_afr/    AFR-Net-style score matrices (FVC2002 DB1, DB3)
experiments/*.json|csv   summary metrics, operation counts, timing benchmark
reports/                 final report (PDF, HTML, LaTeX) and figures
```

## Reproducing the experiment

1. **Environment**: `pip install -r requirements.txt` (CPU is sufficient; a GPU is faster).
2. **Data** (not redistributed): the FVC2002 and FVC2004 databases are distributed with *Handbook of Fingerprint Recognition*, 3rd ed. (Maltoni, Maio, Jain, Feng; Springer 2022), chapter 4 supplementary material, as linked from the [FVC2004 download page](http://bias.csr.unibo.it/fvc2004/download.asp). Extract to `data/raw/` so that `data/raw/74034_3_En_4_MOESM1_ESM/FVC2002/Dbs/Db1_a/1_1.tif` exists.
3. **Models**: clone the official JIPNet repository (MIT licence) into `ext/JIPNet` (`git clone https://github.com/XiongjunGuan/JIPNet ext/JIPNet`; experiments used commit `40d8445`) and download the authors' weights into `ext/JIPNet/ckpts/` from the link in its README (JIPNet, AFRNet, PFVNet AlignNet, RidgeNet).
4. **Run** (from the repository root):

| Step | Command | Output |
|---|---|---|
| Check the models reproduce the authors' example outputs | `python src/check_examples.py` | `experiments/step1_examples.json` |
| Build partial-print pairs | `python src/make_pairs.py` then `python src/patch_coverage.py` | `data/partial/*_pairs.csv` |
| Check ground-truth pose | `python src/check_pairs.py` | `experiments/step2_checks.json` |
| Run JIPNet on all pairs | `python src/run_pairs.py JIPNet 2002_DB1 2002_DB2 …` | `experiments/runs/` |
| Run the AFR-Net-style baseline | `python src/run_afr_embed.py 2002_DB1 2002_DB3` | `experiments/runs_afr/` |
| Timing and operation counts | `python src/bench_compare.py`, `python src/count_flops.py` | `experiments/bench.json`, `flops.json` |
| Metrics and figures | `python src/report_figs.py`, `python src/afr_analysis.py` | `experiments/summary.csv`, `reports/figures/` |
| Report | `python src/build_submission.py`, `python src/build_latex.py` | `reports/` |

The pair definitions and all scores used in the report are included, so the metrics, figures and report can be regenerated (`report_figs.py` onwards) without re-running the models; the image galleries require the FVC data.

## Notes

- Genuine pairs are cut from a single impression with a known shift and rotation (exact ground-truth pose); results are an upper bound for pairs from different captures.
- The AFR-Net-style baseline uses the AFR-Net weights released by the JIPNet authors, used as a fixed-length template.
- Measured timings refer to a 2-core Intel Xeon CPU (2.1 GHz) without GPU; operation counts (GFLOP) are hardware-independent.
