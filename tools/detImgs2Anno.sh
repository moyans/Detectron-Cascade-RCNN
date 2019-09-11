#!/bin/bash

#cuda10: docker pull moyans/detectron_api:v1.2
#cuda9: docker pull moyans/detectron_api:v1.2_cuda9

if [ ! -n "$1" ]; then
    echo Please Enter 1st parm1 model path !!!
    exit
fi
if [ ! -n "$2" ]; then
    echo Please Enter 2st parm1 model config path !!!
    exit
fi
if [ ! -n "$3" ]; then
    echo Please Enter 3st parm1 det_imgs path !!!
    exit
fi
if [ ! "$4" ]; then
    echo  Please Enter 4st parm1 GPU ID !!!
    exit
fi



echo "model path: $1"
echo "model config path : $2"
echo "det img path : $3"
echo "gpu id : $4"

# -e LANG=C.UTF-8 解决docker环境中文乱码
nvidia-docker run -it -e LANG=C.UTF-8 --shm-size 28G --rm \
	-v $PWD:/home/moyan/workdir \
	 moyans/detectron_api:v1.2_cuda9 \
	 /root/.pyenv/versions/idt/bin/python det2xml_with_abspath.py --m $1  --c $2 --t $3 --gpu_id $4
