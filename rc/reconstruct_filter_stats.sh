#!/bin/bash -l

source /home/vn1747/.conda/etc/profile.d/conda.sh
conda activate py38

# Train cgane with different hyper param
gamma_edge_list=(
  "0"
  "0.1"
  "1.5"
  "60"
)

gamma_var_list=(
  "0"
  "0.1"
  "1.5"
  "60"
)

gamma_mean_list=(
  "0"
  "0.1"
  "1.5"
  "60"
)

use_sia_filter_list=(
  false
  true
)


for gamma_edge in "${gamma_edge_list[@]}"
do
  for gamma_mean in "${gamma_mean_list[@]}"
  do
    for gamma_var in "${gamma_var_list[@]}"
    do
      for use_sia_filter in "${use_sia_filter_list[@]}"
      do

        raw_name="srcgan_${gamma_edge//./}_${gamma_mean//./}_${gamma_var//./}"

        if $use_sia_filter
        then
          exp_name="refi_${raw_name}_sia"
          str="#!/bin/bash\npython reconstruct_filter_stats.py \
            -l logs/dda_pipeline/cgan_export \
            -e ${exp_name} \
            --target_path logs/dda_pipeline/cgan_export/${exp_name}/cycled.h5 \
            --model_directory logs/dda_pipeline/srcgan/${raw_name}/checkpoints \
            --source_domain /home/vn1747/data/rit_eyes.h5 \
            --batch_size 2 \
            --sia_filter \
            --target_domain /home/vn1747/data/open_eds_real.h5 \
            --sia_weights_path logs/dda_pipeline/siamese/sia_4/checkpoints/sia_4.pth \
          "
        else
          exp_name="refi_${raw_name}"
          str="#!/bin/bash\npython reconstruct_filter_stats.py \
            -l logs/dda_pipeline/cgan_export \
            -e ${exp_name} \
            --target_path logs/dda_pipeline/cgan_export/${exp_name}/cycled.h5 \
            --model_directory logs/dda_pipeline/srcgan/${raw_name}/checkpoints \
            --source_domain /home/vn1747/data/rit_eyes.h5 \
            --batch_size 2 \
            --target_domain /home/vn1747/data/open_eds_real.h5 \
            --sia_weights_path logs/dda_pipeline/siamese/sia_4/checkpoints/sia_4.pth \
          "
        fi

        echo ${str}
        echo -e $str > command.lock
        sbatch -J ${exp_name} -o "/home/vn1747/vietlib/history/%x_%j.out" -e "/home/vn1747/vietlib/history/%x_%j.err" --ntasks=1 --mem-per-cpu=32g -p tier3 --account=eyeseg --gres=gpu:a100:1 --time=1-00:00:00 command.lock
        rm command.lock
      done
    done
  done
done