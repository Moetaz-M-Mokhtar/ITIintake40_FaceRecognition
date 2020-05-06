FROM ubuntu
# FROM nvidia/cuda:10.2-cudnn7-devel

#application port
EXPOSE 5000

#Download pacjages
RUN apt update -- && apt install -y \
    python \
    git \
    python3-pip \
    python3-setuptools \
    libsm6 \
    libxext6 \
    libxrender1

# Install pip requirements
ADD requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /root/

WORKDIR /root/

ENTRYPOINT  python3 app.py
           

