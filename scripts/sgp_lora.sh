

CUDA_VISIBLE_DEVICES=1 python3 -u main.py --dataset "cars196_224" --smart_defaults --weight_temp 1.0 --vit_type "vit-b-p16" &

sleep 15

CUDA_VISIBLE_DEVICES=4 python3 -u main.py --dataset "cifar100_224" --smart_defaults --weight_temp 1.0 --vit_type "vit-b-p16"&

sleep 15

CUDA_VISIBLE_DEVICES=5 python3 -u main.py --dataset "imagenet-r" --smart_defaults --weight_temp 1.0 --vit_type "vit-b-p16"&

sleep 15

CUDA_VISIBLE_DEVICES=0 python3 -u main.py --dataset "cub200_224" --smart_defaults --weight_temp 1.0 --vit_type "vit-b-p16"&

wait

CUDA_VISIBLE_DEVICES=1 python3 -u main.py --dataset "cars196_224" --smart_defaults --weight_temp 2.0 --vit_type "vit-b-p16" &

sleep 15

CUDA_VISIBLE_DEVICES=4 python3 -u main.py --dataset "cifar100_224" --smart_defaults --weight_temp 2.0 --vit_type "vit-b-p16"&

sleep 15

CUDA_VISIBLE_DEVICES=5 python3 -u main.py --dataset "imagenet-r" --smart_defaults --weight_temp 2.0 --vit_type "vit-b-p16"&

sleep 15

CUDA_VISIBLE_DEVICES=0 python3 -u main.py --dataset "cub200_224" --smart_defaults --weight_temp 2.0 --vit_type "vit-b-p16"&
