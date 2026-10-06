# Contributing to QuadVerdict

Thank you for your interest in contributing to **QuadVerdict**! We welcome contributions from researchers, machine learning practitioners, financial risk analysts, and software engineers.

---

## Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to [abdulhayykhan.dev@gmail.com](mailto:abdulhayykhan.dev@gmail.com).

---

## How Can You Contribute?

You can contribute to QuadVerdict in many ways:
- **Reporting Bugs:** Identifying edge cases, documentation errors, or benchmark discrepancies.
- **Methodological Enhancements:** Adding new statistical hypothesis tests, loss curves, or calibration metrics.
- **Visual & UI Enhancements:** Improving dashboard interactivity, mobile responsiveness, or chart accessibility.
- **Adding New Models:** Implementing new baseline classification paradigms (e.g. XGBoost, LightGBM, CatBoost, Neural Networks) adhering to the strict zero-leakage pipeline protocol.

---

## Development Setup

### 1. Fork & Clone
```bash
git clone https://github.com/<your-username>/QuadVerdict.git
cd QuadVerdict
```

### 2. Virtual Environment
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Local Installation
Run the complete test suite to ensure all unit, invariant, and leakage tests pass:
```bash
pytest
```

---

## Critical Contribution Rules

### 1. Zero Data Leakage Imperative
Any new feature transformation, encoder, or scaler **MUST** be encapsulated inside a `Pipeline` or `ColumnTransformer`. Never apply scaling, imputing, or synthetic oversampling on the entire dataset prior to cross-validation splits.

### 2. Determinism and Random State
All stochastic algorithms, resamplers, and cross-validation partitioners must receive deterministic random seeds (`random_state=config.SEED` or fold-derived seeds).

### 3. Preserving Mathematical Invariants
Any modifications to metric calculation or threshold sweeping must preserve confusion matrix conservation invariants:
- $\text{TP}(t) + \text{FN}(t) = P$
- $\text{FP}(t) + \text{TN}(t) = N$
- Monotonicity of cumulative threshold counts.

### 4. Client Bundle Budget
The standalone web dashboard (`dist/index.html`) must remain **under 1.0 MB** and operate with **zero external runtime JavaScript dependencies**.

---

## Coding Standards & Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and code formatting:

```bash
# Check for linting issues
ruff check .

# Automatically fix linting issues
ruff check --fix .

# Check code formatting
ruff format --check .

# Auto-format all code
ruff format .
```

- Adhere to **PEP 8** style guidelines.
- Provide descriptive docstrings and type annotations for all new public functions and classes.
- Maximum line length is 100 characters.

---

## Submitting Pull Requests

1. **Branch Naming:**
   - `feature/add-xgboost-pipeline`
   - `fix/calibration-curve-axis`
   - `docs/clarify-significance-math`
2. **Commit Messages:**
   - Write clear, concise commit messages following standard conventions:
     - `feat: add CatBoost pipeline with inner CV tuning`
     - `fix: resolve SVG label clipping on small screens`
     - `docs: expand Nadeau-Bengio mathematical derivation`
3. **Pull Request Checklist:**
   - [ ] Automated tests pass (`pytest` passes 100%).
   - [ ] Linter passes (`ruff check .` and `ruff format --check .`).
   - [ ] Any new public methods or CLI flags are documented in `docs/` and `README.md`.
   - [ ] If changing `web/index.html`, recompile `dist/index.html` via `python web/build_site.py --results results/results.json`.

---

## Questions and Support

Feel free to open an issue on GitHub or reach out to the project maintainer at [abdulhayykhan.dev@gmail.com](mailto:abdulhayykhan.dev@gmail.com).
