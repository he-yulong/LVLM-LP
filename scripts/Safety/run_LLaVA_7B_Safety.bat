@echo off
REM Run LLaVA-7B on MMSafety dataset with different prompts (Windows version, single GPU)

REM Set GPU to use (0 = first GPU)
set CUDA_VISIBLE_DEVICES=0

REM Make sure output directories exist
if not exist output\LLaVA-7B (
    mkdir output\LLaVA-7B
)
if not exist output\tmp (
    mkdir output\tmp
)

REM ---------- Run with prompt: mq ----------
echo Running with prompt: mq
python run_model.py ^
  --model_name LLaVA-7B ^
  --model_path extern/liuhaotian/llava-v1.5-7b ^
  --split SD_TYPO ^
  --dataset MMSafety ^
  --prompt mq ^
  --theme safety ^
  --answers_file output\tmp\1_0.jsonl ^
  --num_chunks 1 ^
  --chunk_idx 0 ^
  --temperature 0.0 ^
  --top_p 0.9 ^
  --num_beams 1

copy /Y output\tmp\1_0.jsonl output\LLaVA-7B\Safety_mq.jsonl


REM ---------- Run with prompt: oe ----------
echo Running with prompt: oe
python run_model.py ^
  --model_name LLaVA-7B ^
  --model_path extern/liuhaotian/llava-v1.5-7b ^
  --split SD_TYPO ^
  --dataset MMSafety ^
  --prompt oe ^
  --theme safety ^
  --answers_file output\tmp\1_0.jsonl ^
  --num_chunks 1 ^
  --chunk_idx 0 ^
  --temperature 0.0 ^
  --top_p 0.9 ^
  --num_beams 1

copy /Y output\tmp\1_0.jsonl output\LLaVA-7B\Safety_oe.jsonl


REM ---------- Run with prompt: oeh ----------
echo Running with prompt: oeh
python run_model.py ^
  --model_name LLaVA-7B ^
  --model_path extern/liuhaotian/llava-v1.5-7b ^
  --split SD_TYPO ^
  --dataset MMSafety ^
  --prompt oeh ^
  --theme safety ^
  --answers_file output\tmp\1_0.jsonl ^
  --num_chunks 1 ^
  --chunk_idx 0 ^
  --temperature 0.0 ^
  --top_p 0.9 ^
  --num_beams 1

copy /Y output\tmp\1_0.jsonl output\LLaVA-7B\Safety_oeh.jsonl

echo Done!
