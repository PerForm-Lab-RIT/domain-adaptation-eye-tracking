#!/bin/bash -l

source /home/vn1747/.conda/etc/profile.d/conda.sh
conda activate py38

modelritnet=(
  true
  false
)

src_domain_name_list=(
  "rit_eyes" # original riteyes dataset
  "orig_cgan" # original cgan
  "cycled_synth" # srcgan only
  "cycled_synth_filter" # srcgan with siamese filter
)

n_real_img_list=(
  "64"
  "8192"
)


# new experimental deisgn
n_synth_img_list=(
  "64"
  "256"
  "1024"
  "2048"
  "4096"
)

declare -A epoch_map
epoch_map["64r64s"]="400"
epoch_map["64r256s"]="150"
epoch_map["64r1024s"]="120"
epoch_map["64r2048s"]="100"
epoch_map["64r4096s"]="70"

epoch_map["8192r64s"]="120"
epoch_map["8192r256s"]="100"
epoch_map["8192r1024s"]="80"
epoch_map["8192r2048s"]="70"
epoch_map["8192r4096s"]="60"


# Train every mix dataset model
# for fold in {0..9}
for fold in {1..3}
do
  for src_domain_name in "${src_domain_name_list[@]}"
  do
    for n_real_img in "${n_real_img_list[@]}"
    do
      for n_synth_img in "${n_synth_img_list[@]}"
      do
        for isritnet in "${modelritnet[@]}"
        do
          m="${n_real_img}r${n_synth_img}s"

          if $isritnet
          then
            exp_name="ritnet_mix_${src_domain_name}_${n_real_img}r_${n_synth_img}s"
            str="#!/bin/bash\npython train_ritnet_mix.py \
              -l logs/dda_pipeline/pop2 \
              -e ${exp_name} \
              --n_folds 10
              --fold ${fold}
              --n_real_limit ${n_real_img}
              --n_synth_limit ${n_synth_img}
              --source_domain /home/vn1747/data/${src_domain_name}.h5
              --target_domain /home/vn1747/data/open_eds_real.h5
              --batch_size 8
              --epochs ${epoch_map[${m}]}
            "
          else
            exp_name="dann_mix_${src_domain_name}_${n_real_img}r_${n_synth_img}s"
            str="#!/bin/bash\npython train_dann_mix.py \
              -l logs/dda_pipeline/pop2 \
              -e ${exp_name} \
              --n_folds 10
              --fold ${fold}
              --n_real_limit ${n_real_img}
              --n_synth_limit ${n_synth_img}
              --source_domain /home/vn1747/data/${src_domain_name}.h5
              --target_domain /home/vn1747/data/open_eds_real.h5
              --batch_size 8
              --epochs ${epoch_map[${m}]}
            "
          fi
          echo ${str}
          echo -e $str > command.lock
          sbatch -J ${exp_name}_f${fold} -o "/home/vn1747/vietlib/history/%x_%j.out" -e "/home/vn1747/vietlib/history/%x_%j.err" --ntasks=1 --mem-per-cpu=32g -p tier3 --account=eyeseg --gres=gpu:a100:1 --time=5-00:00:00 command.lock
          rm command.lock
        done
      done
    done
  done
done

