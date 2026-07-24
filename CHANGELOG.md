# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0b5] - 2026-07-24

### Changed
- Default `silhouette_power` is now 0.5 (was 3), matching the configuration used
  for the published results.

### Fixed
- The ring example reproduction notebook now matches the original run exactly:
  ring geometry, sampler seeding, and analysis parameters.

## [0.1.0b3] - 2026-07-24

First public release (beta), published to PyPI.

### Added
- Initial release of the topological-representation and scoring methodology
  from Ashoff (2026) as a standalone, in-memory, sklearn-style library.
- `compare()`: takes a list of embedding arrays, returns an all-pairs
  `ComparisonResult` (score matrix plus optional p-values).
- `TopologicalTransformer` (`fit_transform`) producing a `Representation` with
  persistence diagrams, landscapes, silhouettes, bootstrap replicates, and
  confidence bands.
- Swappable diagram transforms (`RotateScale`, `H0Rotate`, `Identity`), with
  the rotation angle also exposed as a top-level `theta` argument.
- `Scorer` and `score_matrix` with energy statistics and curve metrics, each
  with permutation p-values.
- Optional `[viz]` extra with figure-returning plotters, including the baseline
  report figure from the dissertation.
