FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# 비대화형 모드 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 시스템 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 설치
RUN apt-get update && \
    apt-get install -y gpg-agent && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# pip 설치
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# python3.12를 기본 python3로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# PyTorch 설치 (CUDA 12.1 지원)
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# main.py 실행에 필요한 패키지들 설치
RUN pip install \
    transformers \
    peft \
    datasets \
    accelerate \
    tqdm \
    bitsandbytes

# SSH 설정
RUN mkdir /var/run/sshd && \
    echo 'root:xiirocks' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# SSH 포트 노출
EXPOSE 22

# SSH 서비스 자동 시작
CMD ["/usr/sbin/sshd", "-D"]
