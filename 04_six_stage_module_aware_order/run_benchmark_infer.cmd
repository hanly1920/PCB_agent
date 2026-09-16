@echo off
setlocal EnableExtensions

REM Usage:
REM   run_benchmark_infer.cmd "runs\checkpoints\latest_w1000_k500_bs12_s02.pt" old_latest
REM Run this from the pcb_autoplace_modulev2 project root after extracting benchmarks.zip
REM to .\benchmarks\benchmarks\ .

if "%~1"=="" (
  echo Usage: %~nx0 ^<checkpoint-path^> [run-tag]
  exit /b 2
)

set "CKPT=%~1"
set "TAG=%~2"
if "%TAG%"=="" set "TAG=benchmark_run"

if not exist "%CKPT%" (
  echo Checkpoint not found: %CKPT%
  exit /b 2
)

if not exist "benchmarks\benchmarks\_meta" (
  echo Missing extracted benchmark directory: benchmarks\benchmarks\_meta
  echo Extract benchmarks.zip in the project root so the path becomes:
  echo   .\benchmarks\benchmarks\_meta\
  exit /b 2
)

python scripts\stage_benchmarks.py ^
  --bench_root "benchmarks\benchmarks" ^
  --project_root "." ^
  --out_root "runs\benchmark_inputs"

if errorlevel 1 exit /b 1

REM 16 held-out boards within the model's 114-component training range.
python scripts\step4_infer.py ^
  --test_glob "runs\benchmark_inputs\heldout_inrange\*.json" ^
  --ckpt "%CKPT%" ^
  --out_dir "runs\benchmark_results\%TAG%\heldout_inrange_greedy" ^
  --device cuda ^
  --sequence_policy checkpoint ^
  --layout_preset checkpoint ^
  --beam_width 1 ^
  --beam_topk 16

if errorlevel 1 exit /b 1

python scripts\eval_semantic_metrics.py ^
  --task_glob "runs\benchmark_inputs\heldout_inrange\*.json" ^
  --pred_dir "runs\benchmark_results\%TAG%\heldout_inrange_greedy" ^
  --out "runs\benchmark_results\%TAG%\heldout_inrange_greedy_metrics.json"

echo.
echo Completed held-out in-range benchmark inference.
echo Results: runs\benchmark_results\%TAG%\heldout_inrange_greedy
echo Summary: runs\benchmark_results\%TAG%\heldout_inrange_greedy_metrics.json
endlocal
