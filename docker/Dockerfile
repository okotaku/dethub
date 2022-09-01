FROM nvcr.io/nvidia/pytorch:22.07-py3

RUN apt update -y && apt install -y \
    git
RUN apt-get update && apt-get install -y \
    vim \
    libgl1-mesa-dev

# Install python package.
ADD requirements.txt  /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir openmim==0.3.0 && \
    mim install mmcv==2.0.0rc1 && \
    pip install -r /tmp/requirements.txt && \
    pip uninstall -y opencv-python && pip install opencv-python==4.5.1.48

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US

RUN ln -s /opt/conda/lib/python3.8/site-packages /opt/site-packages
