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


for gamma_edge in "${gamma_edge_list[@]}"
do
  for gamma_mean in "${gamma_mean_list[@]}"
  do
    for gamma_var in "${gamma_var_list[@]}"
    do
      exp_name="srcgan_${gamma_edge//./}_${gamma_mean//./}_${gamma_var//./}"
      str="#!/bin/bash\npython srcgan_train.py \
        -l ../../../history/dda_pipeline/srcgan \
        -e ${exp_name} \
        --source_domain /home/vn1747/data/rit_eyes.h5 \
        --target_domain /home/vn1747/data/open_eds_real.h5 \
        --batch_size 2 \
        --gamma_edge ${gamma_edge} \
        --gamma_mean ${gamma_mean} \
        --gamma_var ${gamma_var} \
      "
      echo ${str}
      echo -e $str > command.lock
      sbatch -J ${exp_name} -o "/home/vn1747/vietlib/history/%x_%j.out" -e "/home/vn1747/vietlib/history/%x_%j.err" --ntasks=1 --mem-per-cpu=32g -p tier3 --account=eyeseg --gres=gpu:a100:1 --time=1-00:00:00 command.lock
      rm command.lock
    done
  done
done