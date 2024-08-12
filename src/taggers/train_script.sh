source ~/venvs/cl_dreamscape/bin/activate
wandb sweep --project train_bert_pos_tagger src/taggers/tagger_sweep_config.yaml


source ~/venvs/cl_dreamscape/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
NUM=1
SWEEP_ID="6h4052mx"
CUDA_VISIBLE_DEVICES=2 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_EVALUATE_OFFLINE=0 wandb agent --count $NUM simpleparadox/train_bert_pos_tagger/$SWEEP_ID

# CUDA_VISIBLE_DEVICES=3 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_EVALUATE_OFFLINE=0 python src/taggers/reimplementation_kaggle_bert_pos_tagging.py --batch_size 512 --num_train_epochs 1 --seed 0 --test_run False