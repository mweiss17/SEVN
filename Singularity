#This is a dockerfile that sets up a full SEVN install
Bootstrap: docker

# Here we ll build our container upon the pytorch container
From: pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

# Export global environment variables
%environment
        # export PATH="/usr/local/anaconda3/bin:$PATH"
        export SHELL=/bin/bash
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/bin
        export PATH=/Gym/gym/.tox/py3/bin:$PATH

# Then we put everything we need to install
%post
        apt -y update && \
        apt install -y \
        python3-setuptools \
        python3-dev \
        python3-opengl \
        libjpeg-dev \
        libboost-all-dev \
        libsdl2-dev \
        libosmesa6-dev \
        patchelf \
        ffmpeg \
        xvfb \
        wget \
        git \
        unzip && \
        apt clean && \
        rm -rf /var/lib/apt/lists/*

        # export env vars
        . /environment

        # get pip
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py

        # create default mount points
        echo "Creating mount points"
        mkdir /scratch
        mkdir /data
        mkdir /dataset
        mkdir /tmp_log
        mkdir /final_log

        pip install gym tensorflow pandas

        # Download Gym
        git clone https://github.com/openai/gym.git || true && \
        cd gym
        pip install -e .
        cd ..

        # install SEVN and dependencies
        # cd /usr/local/
        git clone https://github.com/openai/baselines.git
        cd baselines
        pip install -e .
        cd ..
        git clone https://github.com/mweiss17/SEVN.git
        cd SEVN
        pip install -e .
        cd ..

        # Install pytorch-a2c-ppo-acktr
        git clone https://github.com/simonchamorro/pytorch-a2c-ppo-acktr-gail.git
        cd pytorch-a2c-ppo-acktr-gail
        pip install -e .
        # pip install -r requirements.txt

%runscript
        exec /bin/bash "$@"
