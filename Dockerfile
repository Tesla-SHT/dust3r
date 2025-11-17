FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL description="Docker container for DUSt3R with dependencies installed. CUDA VERSION"
ENV DEVICE="cuda"
ENV MODEL="DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1

# Install apt packages
RUN sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    sed -i "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    rm -f /etc/apt/sources.list.d/* &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends \
        # Determined AI requirements and common tools
        autoconf automake autotools-dev build-essential ca-certificates \
        make cmake ninja-build pkg-config g++ ccache yasm \
        ccache doxygen graphviz plantuml \
        daemontools krb5-user ibverbs-providers libibverbs1 \
        libkrb5-dev librdmacm1 libssl-dev libtool \
        libnuma1 libnuma-dev libpmi2-0-dev \
        openssh-server openssh-client pkg-config nfs-common \
        ## Tools
        git curl wget unzip nano net-tools sudo htop iotop \
        cloc rsync screen tmux xz-utils software-properties-common \
    && rm /etc/ssh/ssh_host_ecdsa_key \
    && rm /etc/ssh/ssh_host_ed25519_key \
    && rm /etc/ssh/ssh_host_rsa_key \
    && cp /etc/ssh/sshd_config /etc/ssh/sshd_config_bak \
    && sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
    && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
    && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install Determined AI stuff
#! ---EDIT notebook-requirements.txt TO ADD PYPI PACKAGES----
WORKDIR /tmp
ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0
ENV JUPYTER_CONFIG_DIR=/run/determined/jupyter/config
ENV JUPYTER_DATA_DIR=/run/determined/jupyter/data
ENV JUPYTER_RUNTIME_DIR=/run/determined/jupyter/runtime
RUN git clone https://github.com/LingzheZhao/determinedai-container-scripts &&\
    cd determinedai-container-scripts &&\
    git checkout v0.2.3 &&\
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple &&\
    pip install determined && pip uninstall -y determined &&\
    pip install -r notebook-requirements.txt &&\
    pip install -r additional-requirements.txt &&\
    jupyter labextension disable "@jupyterlab/apputils-extension:announcements" &&\
    ./add_det_nobody_user.sh &&\
    ./install_libnss_determined.sh &&\
    rm -rf /tmp/*


WORKDIR /
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --recursive https://github.com/naver/dust3r /dust3r
WORKDIR /dust3r
RUN pip install -r requirements.txt
RUN pip install -r requirements_optional.txt
RUN pip install opencv-python==4.8.0.74
RUN python -m pip install --upgrade pip
RUN pip install xformer==0.0.24
RUN pip install torchvision==0.17.0

WORKDIR /dust3r/croco/models/curope/
RUN python setup.py build_ext --inplace

WORKDIR /dust3r
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
#ENTRYPOINT ["/entrypoint.sh"]