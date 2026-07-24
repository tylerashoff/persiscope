# persiscope

Topological representations and pairwise comparison of embeddings. The name is
persistence + periscope.

persiscope turns embedded data into persistence-based summaries (persistence
landscapes and silhouettes) and scores how topologically similar two embedding
sets are. It implements the methodology developed in Ashoff (2026); see the
Reference section below. Everything runs in memory through a sklearn-style
API, with no configuration files and no hidden disk writes.

## Install

```bash
pip install persiscope                 # core
pip install "persiscope[viz]"          # + matplotlib / plotly plotting
pip install "persiscope[metrics]"      # + DTW and Fréchet curve metrics
pip install "persiscope[frame]"        # + pandas for .to_frame()
pip install "persiscope[all]"          # everything above
pip install "persiscope[reproduce]"    # everything above + jupyter, for the dissertation notebooks
```

Requires Python ≥ 3.10. Core dependencies: `numpy`, `scipy`, `networkx`,
`gudhi`, `tqdm`.

## Quickstart

`compare` is the main entry point. It takes a list of embedding arrays and
returns an all-pairs comparison:

```python
import numpy as np
import persiscope as ps

embeddings = [emb_gpt2, emb_bert, emb_t5]   # each (n_i, d)

result = ps.compare(
    embeddings,
    method="energy",        # "energy" or a curve metric ("euclidean", "js", ...)
    summary="landscape",    # "landscape" or "silhouette"
    homology_dim=0,
    theta=-3 * np.pi / 8,   # diagram rotation, the method's nonstandard knob (this is the default)
    n_bootstrap=100,
    run_pvalue=False,
    labels=["gpt2", "bert", "t5"],
    random_state=0,
)

result.matrix               # (3, 3) symmetric dissimilarity matrix
result.to_frame()           # labeled pandas DataFrame (needs [frame] extra)
result.representations      # the fitted Representations, reused across all pairs
```

## The pipeline, step by step

`compare()` chains three stages that you can also drive directly.

### 1. Embeddings to topological representation

Build a distance graph, compute Vietoris–Rips persistence, and summarize it as
landscapes and silhouettes. A subsampling bootstrap makes the result stable and
provides confidence bands.

```python
tf = ps.TopologicalTransformer(
    homology_dim=0,
    diagram_transform=ps.RotateScale(theta=-3 * np.pi / 8, alpha=np.sqrt(2) / 2),
    n_bootstrap=100,
    silhouette_power=0.5,
    tenting_resolution=1000,
    random_state=0,
)
rep = tf.fit_transform(emb_gpt2)             # or distance_matrix=..., or graph=...

rep.mean_landscape          # (resolution, 2) bootstrap-mean curve
rep.bootstrapped_landscapes # (n_bootstrap, resolution, 2)
rep.landscape_band          # {"lower_bound", "upper_bound", "half_bound"}
rep.persistence_diagram     # (n, 2), includes the essential class
```

### 2. Pairwise scoring

Compare two representations, with an optional permutation p-value.

```python
scorer = ps.Scorer(method="energy", summary="landscape", run_pvalue=True, n_permutations=100)
res = scorer.score(rep_a, rep_b)             # ScoreResult(score, pvalue, perm_scores)

matrix = ps.score_matrix([rep_a, rep_b, rep_c], method="energy")
```

### 3. Visualization (optional `[viz]` extra)

Each helper returns a figure object; you decide whether to show or save it.

The main figure is the baseline report. It compares every model to a
reference: overlaid mean landscapes and silhouettes with 95% confidence bands,
the energy statistic against the baseline for each summary type with
permutation-significance stars, and Wasserstein/JS distance strips annotated
the same way.

```python
fig = ps.viz.plot_baseline_report(
    result.representations,
    baseline=0,                          # index of the reference model
    curve_metrics=("wasserstein", "js"),
    n_permutations=200,
)
```

Individual plotters cover the pieces:

```python
ps.viz.plot_persistence_diagram(rep)         # matplotlib Figure
ps.viz.plot_landscape(rep)                   # matplotlib Figure, with band shading
ps.viz.plot_score_heatmap(result)            # plotly Figure
ps.viz.plot_comparison_report(result.representations)   # all-pairs report figure
```

## Diagram transforms

The rotation applied to a persistence diagram before summaries are built is
the nonstandard part of the method, so it is exposed directly alongside
`homology_dim`:

```python
rep = ps.TopologicalTransformer(homology_dim=0, theta=-3 * np.pi / 8).fit_transform(emb)
res = ps.compare(embeddings, theta=-np.pi / 4)   # the conventional rotation, for contrast
```

If you need more than an angle, pass a full transform object instead:
`ps.RotateScale`, `ps.H0Rotate`, `ps.Identity`, or your own implementation of
the small `DiagramTransform` protocol. The rest of the pipeline does not care
which one you use.

```python
from persiscope import transforms

tf = ps.TopologicalTransformer(diagram_transform=transforms.H0Rotate(angle=3 * np.pi / 8))
```

## Scoring methods

| method | family | compares |
| --- | --- | --- |
| `energy` | distribution | the two bootstrap sets of curves (energy distance) |
| `euclidean`, `cosine`, `spectral`, `chi_squared`, `kl`, `js`, `wasserstein` | curve | the two mean curves |
| `dtw`, `frechet` | curve | the two mean curves (needs `[metrics]` extra) |

Larger scores mean more different. The permutation test is one-sided.

## Development

```bash
pip install -e ".[viz,metrics,frame,dev]"
pytest
```

During extraction, persiscope's landscapes and silhouettes were verified to
match the dissertation's original implementation to numerical tolerance.

## Reference

The methodology implemented here (persistence landscapes and silhouettes over
embedding graphs, the rotated-diagram construction, and pairwise scoring with
permutation tests) is developed in:

> Ashoff, T. (2026). *Persistent Convolution: A Topological Approach to Formal
> AI Alignment Testing* (PhD dissertation, University of Virginia).
> DOI: [10.18130/8k9j-9k42](https://doi.org/10.18130/8k9j-9k42)

If you use persiscope in academic work, please cite the dissertation (see
[CITATION.cff](CITATION.cff)).

### Reproducing the dissertation results

Notebooks reproducing the dissertation's results live in
[`examples/reproduction/`](examples/reproduction/) in this repository. Extras
install dependencies, not files, so clone the repository to get the notebooks:

```bash
git clone https://github.com/tylerashoff/persiscope.git
cd persiscope
pip install -e ".[reproduce]"
jupyter notebook examples/reproduction/
```

Reproduced figures are statistically identical to the published ones but not
pixel-identical: the bootstrap uses different random draws than the original
pipeline.

## License

MIT, see [LICENSE](LICENSE).
