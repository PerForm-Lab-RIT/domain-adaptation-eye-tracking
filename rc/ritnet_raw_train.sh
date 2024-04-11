#!/bin/bash -l


source /home/vn1747/.conda/etc/profile.d/conda.sh
conda activate py38

exp_name="ritnet_raw"
str="#!/bin/bash\npython train_ritnet_only.py \
  -l logs/dda_pipeline/pop \
  -e ${exp_name} \
  --dataset_path /home/vn1747/data/open_eds_real.h5 \
  --val_dataset_path /home/vn1747/data/open_eds_real.h5 \
  --batch_size 8 \
  --use_lr_scheduler
"
echo ${str}
echo -e $str > command.lock
sbatch -J ${exp_name} -o "/home/vn1747/vietlib/history/%x_%j.out" -e "/home/vn1747/vietlib/history/%x_%j.err" --ntasks=1 --mem-per-cpu=32g -p tier3 --account=eyeseg --gres=gpu:a100:1 --time=5-00:00:00 command.lock
rm command.lock