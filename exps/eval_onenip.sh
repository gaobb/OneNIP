export PYTHONPATH=../:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$2


checkpoints_path=../checkpoints
dtd_path=../data/dtd  # Change to your path （dtd）
data_path=../data     # Change to your path  (mvtec, visa, btad, mvtec+btad+visa)
num_encoder=4
num_decoder=4

for dataset in mvtec visa btad mvtec+btad+visa 
do
    for img_size in 224 256 320
    do
        python3 -m torch.distributed.launch --nproc_per_node=$1 --master_port=29518 ../tools/train_val.py \
            -e \
            --config ../configs/onenip_config.yaml  \
            --opts  dataset.image_reader.kwargs.image_dir "${data_path}/${dataset}" \
                    dataset.train.dtd_dir  "${dtd_path}" \
                    dataset.train.meta_file  "../data/${dataset}/train.json" \
                    dataset.test.meta_file  "../data/${dataset}/test.json" \
                    dataset.input_size  "[${img_size}, ${img_size}]"  \
                    net[2].kwargs.num_encoder_layers  ${num_encoder} \
                    net[2].kwargs.num_decoder_layers  ${num_decoder} \
                    saver.save_dir   "${checkpoints_path}/onenip-${dataset}-${num_encoder}-${num_decoder}-${img_size}" 
    done
done
