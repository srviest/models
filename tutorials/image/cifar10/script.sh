
export CUDA_VISIBLE_DEVICES="0"
python cifar10_multi_gpu_train.py --num_gpus 1 --max_steps 100000 > Num_gpus_1_steps_100000.txt
export CUDA_VISIBLE_DEVICES="0,1"
python cifar10_multi_gpu_train.py --num_gpus 2 --max_steps 100000 > Num_gpus_2_steps_100000.txt
export CUDA_VISIBLE_DEVICES="0,1,2"
python cifar10_multi_gpu_train.py --num_gpus 3 --max_steps 100000 > Num_gpus_3_steps_100000.txt
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python cifar10_multi_gpu_train.py --num_gpus 4 --max_steps 100000 > Num_gpus_4_steps_100000.txt
export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
python cifar10_multi_gpu_train.py --num_gpus 5 --max_steps 100000 > Num_gpus_5_steps_100000.txt