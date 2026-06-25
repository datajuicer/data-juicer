FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

LABEL maintainer="data-juicer"
LABEL description="Data-Juicer VLA Pipeline pre-built image with MoGe-2, HaWoR, MegaSaM, PyTorch3D support"

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1

# init conda for non-interactive use
RUN conda init bash
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        build-essential \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir py-data-juicer
RUN pip install --no-cache-dir -U "ray[default]"

RUN git clone --depth 1 https://github.com/ThunderVVV/HaWoR.git /root/.cache/data_juicer/assets/HaWoR

# modify requirements.txt:
#   - remove chumpy (need --no-build-isolation, install separately)
#   - remove aitviewer / moderngl-window / pyrender (OpenGL rendering, no head environment needed)
#   - relax torch-scatter version locking (2.1.2 may be incompatible with PyTorch 2.4.1)
#   - relax numpy version locking (to avoid conflict with PyTorch 2.4.1)
RUN cd /root/.cache/data_juicer/assets/HaWoR \
    && sed -i '/^chumpy/d' requirements.txt \
    && sed -i '/^aitviewer/d' requirements.txt \
    && sed -i '/^moderngl-window/d' requirements.txt \
    && sed -i '/^pyrender/d' requirements.txt \
    && sed -i 's/^torch-scatter==.*/# &/' requirements.txt \
    && sed -i 's/^numpy==.*/numpy/' requirements.txt

RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html

# MoGe-2 camera calibration
RUN pip install --no-cache-dir "moge @ git+https://github.com/microsoft/MoGe.git"

# HaWoR dependencies (install from requirements.txt)
RUN pip install --no-cache-dir -r /root/.cache/data_juicer/assets/HaWoR/requirements.txt
# chumpy needs --no-build-isolation, install separately
RUN pip install --no-cache-dir --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"

RUN pip install --no-cache-dir scipy pyarrow av opencv-contrib-python Pillow
RUN pip install --no-cache-dir openai
RUN pip install numpy==1.26.4
RUN pip install opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84
RUN pip install --upgrade --force-reinstall click

# ===============create mega-sam conda environment================
RUN conda create -n mega-sam --clone base -y

# clone mega-sam repository and patch source code
RUN git clone --recursive https://github.com/mega-sam/mega-sam.git /root/.cache/data_juicer/assets/mega-sam \
    && cd /root/.cache/data_juicer/assets/mega-sam \
    # patch .type() -> .scalar_type() to be compatible with new PyTorch
    && sed -i 's/\.type()/\.scalar_type()/g' \
        base/src/altcorr_kernel.cu \
        base/src/correlation_kernels.cu \
        base/src/droid_kernels.cu \
        base/thirdparty/lietorch/lietorch/src/lietorch_gpu.cu \
        base/thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp \
    # patch setup.py: add L20(8.9) and H20(9.0) CUDA arch support
    && sed -i "/compute_86,code=sm_86/a\                    '-gencode=arch=compute_89,code=sm_89',\n                    '-gencode=arch=compute_90,code=sm_90',"  base/setup.py

# build droid_backends + lietorch (CUDA compile, time-consuming)
RUN cd /root/.cache/data_juicer/assets/mega-sam/base && conda run -n mega-sam python setup.py install

RUN conda run -n mega-sam pip install --force-reinstall pydantic pydantic-core typing-extensions
RUN conda run -n mega-sam pip install numpy==1.26.4
RUN conda run -n mega-sam pip install opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84
RUN conda run -n mega-sam pip install --upgrade --force-reinstall click

RUN conda clean -afy && pip cache purge 2>/dev/null || true

WORKDIR /workspace

ENTRYPOINT ["python"]
