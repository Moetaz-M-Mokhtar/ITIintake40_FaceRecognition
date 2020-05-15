#!/bin/bash

if [ $# != 1 ]; then
    echo "You need to specify whether are you going to use cpu or gpu for build"
else
    echo "Inflating scripts"
    if [ "$1" == gpu ]; then
        cp ./detection/docker/Dockerfile_gpu ./detection/Dockerfile
        cp ./detection/docker/model_handler_gpu.py ./detection/models/detection/retinaface-R50/model_handler.py
        cp ./detection/docker/requirements_gpu.txt ./detection/requirements.txt
        cp ./recognition/docker/Dockerfile_gpu ./recognition/Dockerfile
        cp ./recognition/docker/requirements_gpu.txt ./recognition/requirements.txt
    elif [ "$1" == cpu ]; then
        cp ./detection/docker/Dockerfile_cpu ./detection/Dockerfile
        cp ./detection/docker/model_handler_cpu.py ./detection/models/detection/retinaface-R50/model_handler.py
        cp ./detection/docker/requirements_cpu.txt ./detection/requirements.txt
        cp ./recognition/docker/Dockerfile_cpu ./recognition/Dockerfile
        cp ./recognition/docker/requirements_cpu.txt ./recognition/requirements.txt
    else
        echo "argument can be \'cpu\' or \'gpu\' only"
    fi
fi

gdown -O  20180402-114759.zip --no-cookies https://drive.google.com/uc?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-&export=download
for job in `jobs -p`
do
    wait $job
done
mkdir -p ./recognition/models/recognition/
mkdir -p ./detection/models/detection/retinaface-R50/
wget https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip
unzip 20180402-114759.zip -d ./recognition/models/recognition/
unzip retinaface-R50.zip  -d ./detection/models/detection/retinaface-R50/
mkdir -p ./recognition/data/
rm -rf 20180402-114759.zip retinaface-R50.zip


