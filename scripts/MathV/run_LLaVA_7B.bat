@echo off
REM Run LLaVA-7B on MathVista with a single GPU (Windows version)

REM Set GPU to use (0 = first GPU)
set CUDA_VISIBLE_DEVICES=0

REM Define output file
set OUTPUT_FILE=output\LLaVA-7B\MathV.jsonl

REM Make sure output directory exists
if not exist output\LLaVA-7B (
    mkdir output\LLaVA-7B
)

REM Run the model (no chunking since only 1 GPU)
python run_model.py ^
  --model_name LLaVA-7B ^
  --model_path extern/liuhaotian/llava-v1.5-7b ^
  --split testmini ^
  --dataset MathVista ^
  --answers_file %OUTPUT_FILE% ^
  --num_chunks 1 ^
  --chunk_idx 0 ^
  --prompt oe ^
  --temperature 0.0 ^
  --top_p 0.9 ^
  --num_beams 1
