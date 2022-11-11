FROM nvcr.io/nvidia/pytorch:22.07-py3

RUN apt update -y && apt install -y \
    git
RUN apt-get update && apt-get install -y \
    vim \
    libgl1-mesa-dev
ENV FORCE_CUDA="1"

# Install python package.
WORKDIR /dethub
COPY ./ /dethub
RUN pip install --upgrade pip && \
    pip install --no-cache-dir openmim==0.3.2 && \
    pip install . && \
    pip uninstall -y opencv-python && pip install opencv-python==4.5.1.48 && \
    MMCV_WITH_OPS=1 pip install mmcv==2.0.0rc2 && \
    pip install 'git+https://github.com/facebookresearch/detectron2.git' && \
    pip install git+https://github.com/lvis-dataset/lvis-api.git

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US

RUN git config --global --add safe.directory /workspace

WORKDIR /workspace
