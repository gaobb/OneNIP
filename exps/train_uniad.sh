export PYTHONPATH=../:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$2

checkpoints_path=../checkpoints
data_path=../data     # Change to your path  (mvtec, visa, btad, mvtec+btad+visa)
num_encoder=4
num_decoder=4

for dataset in mvtec visa btad mvtec+btad+visa
    do
    for img_size in 224 256 320
    do
        python3 -m torch.distributed.launch --nproc_per_node=$1 --master_port=29518 ../tools/train_val.py \
            --config ../configs/uniad_config.yaml  \
            --opts  dataset.image_reader.kwargs.image_dir "${data_path}/${dataset}" \
                    dataset.train.meta_file  "../data/${dataset}/train.json" \
                    dataset.test.meta_file  "../data/${dataset}/test.json" \
                    dataset.input_size  "[${img_size}, ${img_size}]"  \
                    net[2].kwargs.num_encoder_layers  ${num_encoder} \
                    net[2].kwargs.num_decoder_layers  ${num_decoder} \
                    saver.save_dir   "${checkpoints_path}/uniad-${dataset}-${num_encoder}-${num_decoder}-${img_size}" 
    done
done