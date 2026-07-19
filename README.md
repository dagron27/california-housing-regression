# California Housing Price Prediction — End-to-End Regression Pipeline

![CI](https://github.com/dagron27/california-housing-regression/actions/workflows/ci.yml/badge.svg)

**Course:** `CSCI 441/541, Neural Networks, Fall 2025`

**Assignment:** `csci441-541-project1`

Note: this repository originally lived under a GitHub Classroom organization
(`github.com/St-Cloud-State/csci441-541-project1-dagron27`); it has since
been moved to this personal GitHub account
(`github.com/dagron27/california-housing-regression`). The Classroom copy
remains the official course submission record and was left untouched.

## Assignment Intent

The assignment's stated objectives were to see what a full machine-learning
project looks like, practice the tools involved, and use Python's
Scikit-Learn framework -- via the classic California Housing Prices project
from *Hands-On Machine Learning* (chapter 2), reimplementing the whole
chapter-2 workflow end to end.

**Confirmed implemented**, per the notebook (see Overview below): data
acquisition and loading, exploratory analysis, multiple train/test split
strategies, data cleaning (including an explicitly flagged deviation --
outlier removal via `IsolationForest`, which is not part of the original
textbook chapter), categorical encoding and feature scaling, custom
Scikit-Learn transformers, a full preprocessing `Pipeline`, training and
cross-validated comparison of `LinearRegression`/`DecisionTreeRegressor`/
`RandomForestRegressor`, hyperparameter tuning via `GridSearchCV`/
`RandomizedSearchCV`, final test-set evaluation (RMSE with a 95% confidence
interval), and persisting/reloading the fitted pipeline with `joblib`. This
covers every stage of the chapter-2 workflow the objectives call for.

## Overview

This is a single Jupyter/Colab notebook (`notebooks/CSCI_441_Project1.ipynb`, 273 cells)
that walks through a complete, textbook-style end-to-end machine learning
project on the **California Housing Prices** dataset (the classic dataset
used in Aurelien Geron's *Hands-On Machine Learning with Scikit-Learn, Keras
& TensorFlow*, chapter 2). The notebook:

- Downloads the housing dataset (a `.tgz` of a CSV) from a public GitHub repo
  and loads it with pandas.
- Explores the data (histograms, scatter plots, a California map overlay,
  correlation analysis, engineered ratio features).
- Compares multiple train/test split strategies: pure random shuffling,
  hash-based ID splitting, and stratified sampling on income category
  (`StratifiedShuffleSplit` / `train_test_split(..., stratify=...)`).
- Cleans the data: compares dropping rows, dropping a column, and median
  imputation (`SimpleImputer`) for missing `total_bedrooms` values; also adds
  an outlier-removal step using `IsolationForest` that the author explicitly
  notes is *not* part of the original textbook and changes downstream values.
- Encodes categorical data (`OrdinalEncoder`, `OneHotEncoder`) and scales
  numeric data (`MinMaxScaler`, `StandardScaler`, log transforms, RBF-kernel
  similarity features).
- Builds custom Scikit-Learn transformers (`StandardScalerClone`,
  `ClusterSimilarity` using `KMeans`) and assembles a full `ColumnTransformer`
  / `Pipeline` preprocessing stack.
- Trains and compares `LinearRegression`, `DecisionTreeRegressor`, and
  `RandomForestRegressor`, using k-fold cross-validation to evaluate each.
- Tunes hyperparameters with `GridSearchCV` and `RandomizedSearchCV`.
- Evaluates the final model on the held-out test set, computes an RMSE and a
  95% confidence interval (via both t-score and z-score methods).
- Saves the final fitted pipeline with `joblib.dump` and demonstrates
  reloading it with `joblib.load`.

## Dependencies

Inferred from the notebook's `import` statements:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn` (`sklearn` — `model_selection`, `impute`, `ensemble`,
  `preprocessing`, `cluster`, `compose`, `base`, `utils.validation`,
  `linear_model`, `tree`, `metrics`, `metrics.pairwise`, `pipeline`,
  `experimental.enable_halving_search_cv`)
- `scipy` (`stats`, `stats.randint/uniform/geom/expon/loguniform`)
- `joblib`
- `packaging`
- Python standard library: `sys`, `pathlib`, `tarfile`, `urllib.request`,
  `zlib`, `inspect`

A `requirements.txt` is now included (`pip install -r requirements.txt`).
(The notebook previously also imported `google.colab` for an unused Drive
mount; that cell has been removed — see Known Issues.)

## Environment Setup

The notebook was authored for Google Colab (see `metadata.colab` in the
`.ipynb` file). The one Colab-specific cell (Drive mounting) has been
removed (see Known Issues below), so the notebook now runs unmodified in a
normal Jupyter environment.

1. Use Python 3.7+ (asserted in-notebook) with a recent `scikit-learn`
   (>= 1.0.1, asserted in-notebook).
2. Install dependencies, e.g.:
   `pip install pandas numpy matplotlib scikit-learn scipy joblib packaging`
3. To run in Google Colab or a local Jupyter/Jupyter Lab environment: run
   all cells top to bottom. All data access is via public HTTPS downloads
   and local `datasets/` and `images/` folders created automatically by the
   notebook.
4. No credentials, API keys, or external service accounts are required to
   run the notebook.

## Continuous Integration

A GitHub Actions workflow (`.github/workflows/ci.yml`) re-executes
`notebooks/CSCI_441_Project1.ipynb` end to end (via `nbconvert --execute`) on every
push, including the `RandomForestRegressor`, `DecisionTreeRegressor`, and
`GridSearchCV`/`RandomizedSearchCV` cells. This is intentional, not just a
smoke test.

This means CI retrains all models from scratch in a fresh environment. Most
of the randomized steps in the notebook (`train_test_split`,
`StratifiedShuffleSplit`, `IsolationForest`, `ClusterSimilarity`/`KMeans`,
`DecisionTreeRegressor`, `RandomForestRegressor`, and `RandomizedSearchCV`)
do pass `random_state=42`, so results should generally be close to
reproducible run-to-run. However, `requirements.txt` does not pin exact
package versions, so CI may install different `scikit-learn`/`numpy`/`scipy`
versions than were used when the committed notebook's outputs were
generated; algorithm or default-parameter changes between library versions
can shift metrics (RMSE, cross-validation scores, etc.) slightly even with a
fixed `random_state`. If a CI run's numbers differ a little from the
committed notebook's saved outputs, that is expected and not a bug.

CI runners are ephemeral: the executed notebook is written to a temporary,
non-committed path outside the checked-out repository (not `--inplace`), so
a CI run never overwrites or modifies the notebook file tracked in git.

## Known Issues

### Dead Code

1. **Commented-out legacy metric calls** (cells 218, 222, 262) — Each
   RMSE calculation is preceded by a commented-out call to
   `mean_squared_error(..., squared=False)`, superseded by
   `root_mean_squared_error(...)`. Left in three separate places as dead
   code from a Scikit-Learn API migration.
   - **Fix-it plan:** Remove the commented lines, or consolidate into a
     single markdown note explaining the API change once, near the first
     occurrence.

2. **Reload cell depends on prior in-session state** (cell 271) — The
   custom `ClusterSimilarity` class is commented out
   (`#class ClusterSimilarity(BaseEstimator, TransformerMixin): #    [...]`)
   with the note "excluded for conciseness." `joblib.load` on the saved
   pipeline requires this class to already be defined in the active kernel
   (from cell 169 earlier in the same run). In a fresh kernel/session, or if
   the `.pkl` file is loaded from a different script, this cell will raise
   `AttributeError`/`ModuleNotFoundError` during unpickling.
   - **Fix-it plan:** Move `ClusterSimilarity` and `column_ratio` into a
     small standalone module (e.g., `transformers.py`) that can be imported
     both when training and when reloading, so the saved model is portable
     outside the original notebook session.

3. **Unused Google Drive mount — Fixed.** The `drive.mount('/content/drive')`
   cell (formerly cell 2) has been removed. It was called at the very start
   of the notebook, but nothing later in the notebook read from or wrote to
   `/content/drive`; all data/image/model I/O uses local relative paths
   (`datasets/`, `images/`, and the working directory). The notebook now
   has 273 cells (was 274) and no remaining references to `google.colab` or
   `drive` anywhere in the source.

4. **Unused data-cleaning exploration branches** (cells 85-89) — Three
   alternative cleaning strategies (`housing_option1`, `housing_option2`,
   `housing_option3`) are created for illustration and never referenced
   again; the notebook proceeds with a separate `SimpleImputer`-based
   approach instead.
   - **Fix-it plan:** Intentional/pedagogical — no change strictly needed,
     but a short markdown note clarifying these are illustrative-only would
     reduce reader confusion about why the variables go unused.

### Security / Risk Findings

1. **Broad Google Drive permission grant, currently unused — Fixed.** The
   former cell 2, `drive.mount('/content/drive')`, has been removed (see
   Dead Code item 3 above).
   - **Original risk scenario:** Mounting Drive grants the Colab runtime
     (and any code that runs afterward in the same session, including any
     accidentally pasted or injected code) read/write access to the user's
     entire personal Google Drive, even though this notebook did not use
     it. This widened the blast radius unnecessarily.
   - **Severity (as found):** Medium (broad scope, but no downstream use
     and no evidence of misuse in this notebook).
   - **Resolution:** The mount cell was deleted entirely rather than left
     dormant, eliminating the unnecessary permission grant.

2. **Unauthenticated, unverified downloads** — Cell 7:
   `urllib.request.urlretrieve(url, tarball_path)` fetching
   `https://github.com/ageron/data/raw/main/housing.tgz`; Cell 64: a similar
   download of `california.png` from
   `https://github.com/ageron/handson-ml3/raw/main/...`.
   - **Risk scenario:** Both URLs point to the `main` branch of a specific
     third-party GitHub repository with no checksum/hash verification of the
     downloaded bytes. If that upstream repository were ever compromised, or
     the URL later repurposed, the notebook would silently download and use
     attacker-controlled content with no integrity check.
   - **Severity:** Low (source is a well-known, fixed public repository, not
     user-supplied input; transport is HTTPS).
   - **Fix-it plan:** Verify a SHA-256 checksum of the downloaded file before
     use, and/or pin the URL to a specific commit SHA instead of `main`.

3. **Tarball extraction without member validation — Fixed.** Cell 7
   (formerly cell 8, before the drive-mount cell removal) called
   `housing_tarball.extractall(path="datasets")` on the downloaded `.tgz`
   with no filter, which on older Python versions could let a malicious
   archive write files outside the target directory via `../` paths in
   member names ("tar path traversal"). The call now explicitly passes the
   safe extraction filter:
   `housing_tarball.extractall(path="datasets", filter="data")`. This
   rejects members whose resolved path would escape the target directory,
   drop absolute paths, and strip unsafe file modes/links, regardless of
   the Python version's default filter setting. `filter="data"` is
   available in Python 3.12+, and was also backported to some 3.8-3.11
   patch releases (PEP 706); it is set explicitly here rather than relying
   on the interpreter's default.

4. **Pickle-based model serialization (`joblib`)** — Cells 270-271:
   `joblib.dump(final_model, "my_california_housing_model.pkl")` and
   `joblib.load("my_california_housing_model.pkl")`.
   - **Risk scenario:** `joblib` uses `pickle` internally; unpickling data
     from an untrusted source can execute arbitrary code. In this notebook
     the file is written and immediately reloaded within the same trusted
     session, so no untrusted deserialization currently occurs. The risk is
     forward-looking: if this `.pkl` artifact is later shared, committed, or
     reloaded from an external/untrusted source, it becomes an arbitrary
     code execution vector.
   - **Severity:** Low (informational; no untrusted deserialization present
     today).
   - **Fix-it plan:** If this model is ever distributed outside the original
     session, treat the `.pkl` file as untrusted input requiring provenance
     verification, or consider a non-pickle serialization format (e.g.,
     ONNX or `skops`) for any model artifact intended for external
     distribution.

No hardcoded credentials, API keys, or tokens; no shell execution (`!`
magics, `os.system`, `subprocess`); and no other external network calls
beyond the two downloads above were found in the notebook.

## Status

Coursework / educational assignment. The notebook appears complete and
runnable end-to-end as a single pass from top to bottom, following the
"Hands-On Machine Learning" chapter 2 workflow with an intentional,
documented deviation (outlier removal via `IsolationForest`) that the author
notes changes the resulting values from the textbook. Not intended for
production use or for processing sensitive/untrusted data as-is; see Known
Issues above before repurposing.
