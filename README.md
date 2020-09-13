Just for learning

Welcome to clone if you are interested.


PPO:
https://github.com/nikhilbarhate99/PPO-PyTorch




==Ubuntu==
```
apt update
apt install software-properties-common
apt install libfreetype6-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev build-essential libssl-dev libffi-dev
add-apt-repository ppa:deadsnakes/ppa
apt install python3.8 python3.8-dev python3.8-venv
python3.8 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

==CUDA==
```
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt update
sudo apt install cuda
```
```
nvidia-smi
```

Tricks:
1. n-Steps Rewards
2. GAE
3. GRU RNN Network 
   1. Bidirectional
   2. 2 layers
   3. 0.2 dropout
4. CNN
5. Reward Normalization instead of Advantage Normalization
6. Orthogonal weights initialization
7. 0 Constant bias initialization
8. ICM (TODO)
    Resources:
    https://zhuanlan.zhihu.com/p/66303476
    https://github.com/bonniesjli/icm/blob/master/icm.py
    https://github.com/jcwleo/curiosity-driven-exploration-pytorch/blob/master/model.py
9. MSTC (TODO) AlphaZero