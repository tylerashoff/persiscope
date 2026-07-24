# Dissertation result reproductions

Notebooks reproducing the results of:

> Ashoff, T. (2026). *Persistent Convolution: A Topological Approach to Formal
> AI Alignment Testing* (PhD dissertation, University of Virginia).
> DOI: [10.18130/8k9j-9k42](https://doi.org/10.18130/8k9j-9k42)

Setup:

```bash
pip install -e ".[reproduce]"   # from the repo root
jupyter notebook examples/reproduction/
```

| notebook | reproduces | data |
| --- | --- | --- |
| `ring_example.ipynb` | Ring example baseline report | synthetic (generated in-notebook, seeded) |

Notes:

- The original figures were produced by the `pleats` research pipeline.
  persiscope matches it to numerical tolerance on the summary functions, but
  bootstrap draws use different random sequences, so reproduced figures are
  statistically identical rather than pixel-identical.
- Results that depend on model embeddings (rather than synthetic data) load
  precomputed embedding artifacts instead of re-running model inference; those
  notebooks document their data provenance individually.
