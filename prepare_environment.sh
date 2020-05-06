#!/bin/bash

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


