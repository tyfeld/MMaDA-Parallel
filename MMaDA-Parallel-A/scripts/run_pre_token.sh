set +e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export PYTHONPATH="$PROJECT_ROOT"
cd "$PROJECT_ROOT"

target_resolution=512
edit_type="your_data_type" # t2i, edit, ti2ti, mmu_single_image, mmu_multi_image
main_name="${edit_type}_parquet"

# replace the variables with your own paths
DATASET_IDS="PATH/TO/YOUR/DATASETS"
# DATASET_IDS="datasetA,datasetB,datasetC"

log_dir="$PROJECT_ROOT/pre_token/${edit_type}/${main_name}-${target_resolution}_log"
out_dir="$PROJECT_ROOT/pre_token/${edit_type}/${main_name}_vae_code-${target_resolution}"

mkdir -p "$log_dir"
mkdir -p "$out_dir"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

GPUS=(0 1 2 3 4 5 6 7)
NGPUS=${#GPUS[@]}
TOTAL_SPLITS=32

for i in $(seq 0 $((TOTAL_SPLITS - 1)))
do
  gpu_id=${GPUS[$((i % NGPUS))]}
  export CUDA_VISIBLE_DEVICES=${gpu_id}

  python3 -u $PROJECT_ROOT/pre_tokenizer/pre_tokenize.py \
    --splits=${TOTAL_SPLITS} \
    --rank=${i} \
    --out_dir "$out_dir" \
    --type ${edit_type} \
    --target_size ${target_resolution} \
    --dataset_ids "${DATASET_IDS}" \
    > "${log_dir}/${target_resolution}-${i}.log" 2>&1 &

  sleep 0.5
done

echo "All processes launched. Waiting for completion..."

wait

finished_count=0
for i in $(seq 0 $((TOTAL_SPLITS - 1)))
do
  progress_file="$out_dir/${i}-of-${TOTAL_SPLITS}-progress.txt"
  if [ -f "$progress_file" ]; then
    status=$(cat "$progress_file")
    if [ "$status" = "finished" ]; then
      ((finished_count++))
    else
      echo "Warning: Rank ${i} did not finish (last index: $status)"
    fi
  else
    echo "Warning: Rank ${i} has no progress file"
  fi
done

echo "Finished ranks: ${finished_count}/${TOTAL_SPLITS}"

if [ ${finished_count} -eq ${TOTAL_SPLITS} ]; then
  python3 -u $PROJECT_ROOT/pre_tokenizer/concat_record.py \
    --sub_record_dir "$out_dir" \
    --save_path "$out_dir/all_records.json"
else
  exit 1
fi
