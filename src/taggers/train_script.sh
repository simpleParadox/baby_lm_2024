source ~/venvs/cl_dreamscape/bin/activate
wandb sweep --project babylm_2024 src/taggers/tagger_sweep_config.yaml


source ~/venvs/cl_dreamscape/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
NUM=1
SWEEP_ID="m3c61e4g"
HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_EVALUATE_OFFLINE=0 CUDA_VISIBLE_DEVICES=4 wandb agent --count $NUM simpleparadox/babylm_2024/$SWEEP_ID
