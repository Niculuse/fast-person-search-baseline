CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port 29504 --nproc_per_node 2 train.py --gpu_id 2,3 \
--dataset prw --batch_size 8 --logs_dir 'logs' --lr 0.009 --backbone resnet50_ibn_a

CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port 29504 --nproc_per_node 2 train.py --gpu_id 2,3 \
--dataset sysu --batch_size 8 --logs_dir 'logs' --lr 0.009 --backbone resnet50_ibn_a

