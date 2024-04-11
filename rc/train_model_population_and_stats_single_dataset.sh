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

# 5000 real as adversarial for dann
declare -A epoch_map
epoch_map["ritnet64s"]="1600" 
epoch_map["ritnet256s"]="800"
epoch_map["ritnet1024s"]="200"
epoch_map["ritnet2048s"]="100"
epoch_map["ritnet4096s"]="70"

epoch_map["dann64s"]="1600" 
epoch_map["dann256s"]="800"
epoch_map["dann1024s"]="200"
epoch_map["dann2048s"]="100"
epoch_map["dann4096s"]="70"


# Train every synth dataset only
for fold in {1..3}
do
  for src_domain_name in "${src_domain_name_list[@]}"
  do
    for n_synth_img in "${n_synth_img_list[@]}"
    do
      for isritnet in "${modelritnet[@]}"
      do
        
        if $isritnet
        then
          m="ritnet${n_synth_img}s"
          exp_name="ritnet_only_${src_domain_name}_${n_synth_img}s"
          str="#!/bin/bash\npython train_ritnet_only.py \
            -l logs/dda_pipeline/pop2 \
            -e ${exp_name} \
            --n_folds 10
            --fold ${fold}
            --dataset_path /home/vn1747/data/${src_domain_name}.h5
            --val_dataset_path /home/vn1747/data/open_eds_real.h5
            --batch_size 8
            --epochs ${epoch_map[${m}]}
            --n_limit ${n_synth_img}
          "
        else
          m="dann${n_synth_img}s"
          exp_name="dann_only_${src_domain_name}_${n_synth_img}s"
          str="#!/bin/bash\npython train_dann_only.py \
            -l logs/dda_pipeline/pop2 \
            -e ${exp_name} \
            --n_folds 10
            --fold ${fold}
            --source_domain /home/vn1747/data/${src_domain_name}.h5
            --target_domain /home/vn1747/data/open_eds_real.h5
            --batch_size 8
            --epochs ${epoch_map[${m}]}
            --n_limit ${n_synth_img}
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