@echo off
rem DeepRBF 3.3.3 benchmark suite for the Computers & Geosciences paper (about 60-90 minutes).
cd /d "%~dp0"
title DeepRBF 3.3.3 benchmarks - do not close
python -c "import deeprbf_recovered, scipy, numpy; print('DeepRBF native', getattr(deeprbf_recovered,'__version__','?'))" || (echo Python environment with DeepRBF not found on PATH & pause & exit /b 1)
if exist results\bench_results.jsonl ren results\bench_results.jsonl bench_results_%RANDOM%.old.jsonl
python bench_deeprbf.py > results_console.txt 2>&1
type results_console.txt
echo.
echo Finished. Results: %~dp0results\bench_results.jsonl .
pause
