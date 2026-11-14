# Benchmarks

All random inputs are seeded. Scripts and raw results behind the performance and accuracy figures in the DeepRBF software paper.

- `bench_deeprbf.py`: RBF engine suite. B0 environment and build flags; B1 interpolation accuracy (3-D Franke
  function, kernels x N); B2 solver scaling (dense direct, GMRES with two-level Schwarz, sparse Cholesky/Schur);
  B3 drift and extrapolation; B4 prediction backends; B5 synthetic two-lens orebody recovery (drill spacing x nugget,
  indicator RBF vs nearest neighbour); B6 the three Kilamongo shell methods (`examples/kilamongo`, or set
  `KILAMONGO_DIR`). `python bench_deeprbf.py B1 B5` runs a subset, `--quick` is a smoke test.
- `bench_geostats.py`: GSLIB-convention geostatistics. G1 `kriging_gslib` against a direct solve, 10^4-10^6
  targets; G2 `sgs_gslib` on 400,000 nodes (histogram and variogram reproduction, single vs multiple grid); G3
  `sis_gslib` on 400,000 nodes; 3.3.3 `run_sgs`/`run_sis` for comparison.
- `RUN_BENCHMARKS.bat`: Windows runner (about 60-90 minutes).
- `results/*.jsonl`: one JSON record per benchmark case, including machine, thread count and library versions;
  `results/_b*.npy`: predicted values used for the accuracy comparisons; `results_console.txt`: console log.

`bench_results.jsonl` was produced with 3.3.3, `bench_results_3_3_4_B2.jsonl` and `bench_geostats.jsonl` with
3.3.4, whose code is identical to 3.4.0. Timings are from one Windows workstation (24 logical CPUs, portable build
without oneMKL or ScalFMM) and will differ on other machines.

```
cd benchmarks
python bench_deeprbf.py
python bench_geostats.py
```
