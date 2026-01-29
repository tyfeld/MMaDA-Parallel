cd "$(dirname "$0")/.."

# replace the variables with your own paths
INIT_FROM="PATH/TO/YOUR/Lumina-DiMOO"
DATA_CONFIG="./configs/data.yaml"
OUTPUT_DIR="PATH/TO/YOUR/OUTPUT_DIR"
mkdir -p ${OUTPUT_DIR}

torchrun \
  --nproc_per_node=8 \
  --master_port=54321 \
  train/train.py \
  --init_from ${INIT_FROM} \
  --data_config ${DATA_CONFIG} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 4 \
  --accum_iter 4 \
  --epochs 200 \
  --lr 2e-5 \
  --min_lr 0.0 \
  --wd 0.1 \
  --warmup_epochs 0.001 \
  --clip_grad 4.0 \
  --max_seq_len 5120 \
  --dropout 0.05 \
  --num_workers 16 \
  --model_parallel_size 1 \
  --data_parallel fsdp \
  --precision bf16 \
  --grad_precision fp32 \
  --save_interval 1 \
  --save_iteration_interval 1000 \
  --ckpt_max_keep 2 \
  --seed 42 \
  --cache_ann_on_disk \
  --pin_mem \
  --checkpointing
