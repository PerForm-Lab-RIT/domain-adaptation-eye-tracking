#!/bin/bash -l

source /home/vn1747/.conda/etc/profile.d/conda.sh
conda activate py38

# example config of structure retaining cycle gan hyperparams
gamma_edge=0.1
gamma_mean=0
gamma_var=60
raw_name="srcgan_${gamma_edge//./}_${gamma_mean//./}_${gamma_var//./}"

# make cycled synth filter
exp_name="refi_export_${raw_name}_sia"
str="#!/bin/bash\npython reconstruct_filter_stats.py \
  -l logs/dda_pipeline/cgan_export \
  -e ${exp_name} \
  --target_path /home/vn1747/data/cycled_synth_filter.h5 \
  --model_directory logs/dda_pipeline/srcgan/${raw_name}/checkpoints \
  --source_domain /home/vn1747/data/rit_eyes.h5 \
  --batch_size 2 \
  --sia_filter \
  --target_domain /home/vn1747/data/open_eds_real.h5 \
  --sia_weights_path logs/dda_pipeline/siamese/sia_4/checkpoints/sia_4.pth \
  --export_h5
"
echo ${str}
echo -e $str > command.lock
sbatch -J ${exp_name} -o "/home/vn1747/vietlib/history/%x_%j.out" -e "/home/vn1747/vietlib/history/%x_%j.err" --ntasks=1 --mem-per-cpu=32g -p tier3 --account=eyeseg --gres=gpu:a100:1 --time=1-00:00:00 command.lock
rm command.lock


# make cycled synth no filter
exp_name="refi_export_${raw_name}"
str="#!/bin/bash\npython reconstruct_filter_stats.py \
  -l logs/dda_pipeline/cgan_export \
  -e ${exp_name} \
  --target_path /home/vn1747/data/cycled_synth.h5 \
  --model_directory logs/dda_pipeline/srcgan/${raw_name}/checkpoints \
  --source_domain /home/vn1747/data/rit_eyes.h5 \
  --batch_size 2 \
  --target_domain /home/vn1747/data/open_eds_real.h5 \
  --sia_weights_path logs/dda_pipeline/siamese/sia_4/checkpoints/sia_4.pth \
  --export_h5
"
echo ${str}
echo -e $str > command.lock
sbatch -J ${exp_name} -o "/home/vn1747/vietlib/history/%x_%j.out" -e "/home/vn1747/vietlib/history/%x_%j.err" --ntasks=1 --mem-per-cpu=32g -p tier3 --account=eyeseg --gres=gpu:a100:1 --time=1-00:00:00 command.lock
rm command.lock


# export original gan also
# make cycled synth no filter
raw_name="srcgan_0_0_0"
exp_name="refi_export_${raw_name}"
str="#!/bin/bash\npython reconstruct_filter_stats.py \
  -l logs/dda_pipeline/cgan_export \
  -e ${exp_name} \
  --target_path /home/vn1747/data/orig_cgan.h5 \
  --model_directory logs/dda_pipeline/srcgan/${raw_name}/checkpoints \
  --source_domain /home/vn1747/data/rit_eyes.h5 \
  --batch_size 2 \
  --target_domain /home/vn1747/data/open_eds_real.h5 \
  --sia_weights_path logs/dda_pipeline/siamese/sia_4/checkpoints/sia_4.pth \
  --export_h5
"
echo ${str}
echo -e $str > command.lock
sbatch -J ${exp_name} -o "/home/vn1747/vietlib/history/%x_%j.out" -e "/home/vn1747/vietlib/history/%x_%j.err" --ntasks=1 --mem-per-cpu=32g -p tier3 --account=eyeseg --gres=gpu:a100:1 --time=1-00:00:00 command.lock
rm command.lock