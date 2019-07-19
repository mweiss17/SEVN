#This is a dockerfile that sets up a full SEVN install
Bootstrap: docker

# Here we ll build our container upon the pytorch container
From: pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

# Export global environment variables
%environment
        export PATH="/usr/local/anaconda3/bin:$PATH"
        export SHELL=/bin/bash
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/bin
        export PATH=/Gym/gym/.tox/py3/bin:$PATH

# Then we put everything we need to install
%post
        apt -y update && \
#        apt install -y keyboard-configuration && \
        apt install -y \
        python3-setuptools \
        python3-dev \
        python-pyglet \
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

        # create default mount points
        echo "Creating mount points"
        mkdir /scratch
        mkdir /data
        mkdir /dataset
        mkdir /tmp_log
        mkdir /final_log

        # download and install Anaconda
        CONDA_INSTALL_PATH="/usr/local/anaconda3"
        wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
        chmod +x Anaconda3-5.0.1-Linux-x86_64.sh
        ./Anaconda3-5.0.1-Linux-x86_64.sh -b -p $CONDA_INSTALL_PATH

        # Create and enter our conda environment
        conda env create -n SENV
        source activate SENV

        # install conda packages
        conda install torchvision -c soumith conda-forge tensorflow

        # Download Gym
        mkdir /Gym && cd /Gym
        git clone https://github.com/openai/gym.git || true && \

        # Install python dependencies
        pip install -r requirements.txt

        # Install Gym
        cd /Gym/gym
        pip install -e '.[all]'

        # install SEVN and dependencies
        cd /usr/local/
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
        pip install -r requirements.txt

%runscript
        exec /bin/bash "$@"
