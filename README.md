# **Human Face Identification**
By *Moetaz Mohamed Mokhtar*

***Still under construction***

## Introduction
In this repo, we provide source codes to achieve deep face recognition to datasets with low number of images per class using KNN classifiers to compare images with the dataset.

## Getting started
These instructions will help you get a running copy of the project.

All the comands assumes you are in **the root folder for this project** so make changes to the paths depending on your file system and current working directory.

if will be working with gpu in your environment you need to set up cuda cores and the gpu drivers as this isn't described in this document.

#### Using Google Colab

You can find the `.ipynb` file in the repo [here](https://github.com/Moetaz-M-Mokhtar/ITIintake40_FaceRecognition/blob/master/Face_Recognition.ipynb)

1. Go to [Google Colab](https://colab.research.google.com/)
2. from file -> open notebook go to github tab and paste the repo link then search.

*Don't forget to set the Hardware acceleration to **GPU***

You will find extra features like data augmentation in the colab file.

#### Using Docker

to run this project using docker you will first need to:

1. clone this repo to your server host
```
$ git clone https://github.com/Moetaz-M-Mokhtar/ITIintake40_FaceRecognition.git
```
2. run the `prepare_environment.sh` as a bash script with a `cpu` or a `gpu` argument depending on your machine and environment
```
$ chmod u+x ./prepare_environment.sh
$ ./prepare_environment.sh cpu
```
3. build the docker files in the detection and in the recognition directories 
```
$ sudo docker build ./detection -t face_detection
$ sudo docker build ./recognition -t face_recognition
 ```
4. run the docker images
```
$ sudo docker run face_detection
$ sudo docker run face_recognition
```
5. send images to the docker server using following requests

METHOD         | Request URL                       | Function
-------------- | :-------------------------------- | -----------
POST           | <base_uri>/recognize              | recognize an image
POST           | <base_uri>/add_face               | adds a user to the recognition dataset

Default value for <base_uri> is http://0.0.0.0:5002/model/api/v1.0/

All requests should be made as application/json

the content of each message is as follow

`recognize:`
```
{
    'img': base64 encoded image 
}
```
`add_face:`
```
{
    'img': base64 encoded image 
    'label': person name / ID
}
```

**Using your personal environment**

clone the repo to your PC
```
$ git clone https://github.com/Moetaz-M-Mokhtar/ITIintake40_FaceRecognition.git
```
Download required packages
```
$ sudo apt update && apt install -y \
    python \
    python3-pip \
    python3-setuptools \
    libsm6 \
    libxext6 \
    libxrender1 \
```
Download required libraries

cpu:
```
$ pip3 install -r ./detection/docker/requirements_cpu.txt
$ pip3 install -r ./recognition/docker/requirements_cpu.txt
```
gpu:
```
$ pip3 install -r ./detection/docker/requirements_gpu.txt
$ pip3 install -r ./recognition/docker/requirements_gpu.txt
```

run the main source code by
```
$ python Face_recognition_main.py ${input_image path} ${output directory} ${face detection model} ${face recognition model} ${Path to KNN features & labels}
```

the output should be the input image with the detect face and the recognized labels.

## Pretrained Models
**Detection models**

| Method           | LFW(%)   | CFP-FP(%) | AgeDB-30(%) | MegaFace(%) | Download  |  
| -------          | ------   | --------- | ----------- |-------------|-----------| 
| LResNet100E      | 99.77    | 98.27     | 98.28       | 98.47       | [baidu cloud](https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA) and [dropbox](https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0) |
| LResNet50E      | 99.80     | 92.74     | 97.76       | 97.64       | [baidu cloud](https://pan.baidu.com/s/1mj6X7MK) and [dropbox](https://www.dropbox.com/s/ou8v3c307vyzawc/model-r50-arcface-ms1m-refine-v1.zip?dl=0) |
| LResNet34E      | 99.65     | 92.12     | 97.70       | 96.70       | [baidu cloud](https://pan.baidu.com/s/1jKahEXw) and [dropbox](https://www.dropbox.com/s/yp7y6jic464l09z/model-r34-arcface-ms1m-refine-v1.zip?dl=0) |
| MobileFaceNet      | 99.50     | 88.94     | 95.91       |------- | [baidu cloud](https://pan.baidu.com/s/1If28BkHde4fiuweJrbicVA) and [dropbox](https://www.dropbox.com/s/akxeqp99jvsd6z7/model-MobileFaceNet-arcface-ms1m-refine-v1.zip?dl=0) |

**Recognition models**

| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) | 0.9905        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

## Face Detection
please check [insightface/RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace) for more info

## Face Recognition
please check [Facenet](https://github.com/davidsandberg/facenet) repository for more info

## License
The source code in this repo and all the sub repos inside it falls under MIT License you can read more about it in the project [LICENSE file](https://github.com/Moetaz-M-Mokhtar/ITIintake40_FaceRecognition/blob/master/LICENSE)

the pretrained models available doesn't fall under the same license and is only available only for non-commercial research purposes only.

# Contact

[Moetaz Mohamed Mokhtar](moetazm.mokhtar@gmail.com)
