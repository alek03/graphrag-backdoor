#!/bin/bash
# Step 0b: index a raw dataset into a knowledge graph (stage1 + stage2).
#
# Builds the KG with a LOCAL, FREE LLM via ollama (qwen2.5:7b) -- NO OpenAI key,
# NO paid API. The KG construction (OpenIE triple extraction + NER) does require
# an LLM, but ollama runs it locally on the GPU.
#
# Prereqs (one time):
#   - install ollama (https://ollama.com), then:  ollama pull qwen2.5:7b
#   - build the raw data first:  python build_raw_dataset.py --dataset all
#
# Usage:
#   sbatch submit_index.sh pilotData
#   sbatch submit_index.sh pilotData2_victim
# (or run the python block directly on any machine with ollama + a GPU)
#
# NOTE: LLM extraction is non-deterministic, so the resulting graph will differ
# slightly from ours -> results are qualitatively similar, not bit-identical.
#SBATCH --job-name=index_kg
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40
#SBATCH --account=open
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=6:00:00
#SBATCH --output=logs/index_%j.out
#SBATCH --error=logs/index_%j.err
set -e

DATA_NAME="${1:-pilotData}"          # which dataset under data/ to index
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"

mkdir -p logs
module load cuda/12.6.0 2>/dev/null || true
module load gcc/12.2.0 2>/dev/null || true
source ~/.bashrc
conda activate gfmrag

# start a local ollama server on the GPU
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/scratch/ollama_models}"
unset CUDA_VISIBLE_DEVICES
ollama serve > ~/ollama.log 2>&1 &
OLLAMA_PID=$!
sleep 40
tail -5 ~/ollama.log || true

python -m gfmrag.workflow.index_dataset \
    dataset.root=data \
    dataset.data_name="${DATA_NAME}" \
    openie_model.llm_api=ollama \
    openie_model.model_name="${OLLAMA_MODEL}" \
    ner_model.llm_api=ollama \
    ner_model.model_name="${OLLAMA_MODEL}" \
    graph_constructor.force=false \
    sft_constructor.force=false

kill $OLLAMA_PID 2>/dev/null || true
