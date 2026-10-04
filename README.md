# QuadVerdict

> **Status: In progress.** This README will be completed after the full benchmark run.

QuadVerdict compares **Logistic Regression, Decision Tree, Random Forest and SVM** on the
UCI *Default of Credit Card Clients* dataset using rigorous methodology: leakage-free pipelines,
nested cross-validation, calibration analysis, statistical significance testing and
**cost-based threshold selection**.

The core thesis is not "which model has the highest accuracy" — it is "which model costs the least
when a missed default is more expensive than a false alarm, and is that difference statistically real?"

---

## Live Demo

*Coming soon — deployed on Vercel after the full benchmark run.*

---

## Reproduction

```bash
# 1. Install dependencies (Python 3.11+)
pip install -r requirements.txt

# 2. Download the dataset
python data/download.py

# 3. Run the benchmark (up to ~4 hours; checkpoints on interrupt)
python run_benchmark.py

# 4. Build the site
python web/build_site.py

# 5. Open dist/index.html in a browser (works offline via file://)
```

### Quick smoke test

```bash
python run_benchmark.py --quick   # ~2 min, tiny budget, not for publication
pytest
```

---

## Results

*Results table, significance summary and methodology details will be added here after Phase 10.*

---

## Limitations

*Written by hand after the full run — placeholder.*

---

## Repository Structure

```
quadverdict/
├── README.md
├── docs/
│   ├── PRD.md              # Product requirements
│   └── TRD.md              # Technical requirements
├── vercel.json
├── requirements.txt
├── pyproject.toml
├── run_benchmark.py        # One-command pipeline entry point
├── data/
│   ├── raw/                # Downloaded CSV (gitignored)
│   └── download.py
├── src/
│   ├── config.py
│   ├── data.py
│   ├── pipelines.py
│   ├── nested_cv.py
│   ├── metrics.py
│   ├── significance.py
│   ├── explain.py
│   ├── experiments.py
│   ├── timing.py
│   └── export.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_pipeline_sanity.ipynb
│   └── 03_results_analysis.ipynb
├── web/
│   ├── index.html          # Source template
│   ├── mock_results.json
│   └── build_site.py
├── dist/
│   └── index.html          # Generated — do not edit by hand
├── results/
│   ├── results.json        # Canonical benchmark output
│   ├── schema.json
│   └── figures/
└── tests/
    ├── test_data.py
    ├── test_pipelines_leakage.py
    ├── test_nested_cv_determinism.py
    ├── test_metrics.py
    ├── test_significance.py
    ├── test_export_schema.py
    └── test_build_site.py
```

---

## Owner

**ABDI (Abdul Hayy Khan)** · BS Artificial Intelligence · Dawood University of Engineering & Technology
