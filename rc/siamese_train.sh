#!/bin/bash -l

source /home/vn1747/.conda/etc/profile.d/conda.sh
conda activate py38

exp_name="sia_4"
str="#!/bin/bash\npython siamese_train.py \
  -l logs/dda_pipeline/siamese \
  -e ${exp_name} \
  --source_domain /home/vn1747/data/rit_eyes.h5 \
  --target_domain /home/vn1747/data/open_eds_real.h5 \
  --batch_size 4 \
"
echo ${str}
echo -e $str > command.lock
sbatch -J ${exp_name} -o "/home/vn1747/vietlib/history/%x_%j.out" -e "/home/vn1747/vietlib/history/%x_%j.err" --ntasks=1 --mem-per-cpu=32g -p tier3 --account=eyeseg --gres=gpu:a100:1 --time=1-00:00:00 command.lock
rm command.lock