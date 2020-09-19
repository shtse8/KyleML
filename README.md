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
8. ICM
    Resources:
    https://zhuanlan.zhihu.com/p/66303476
    https://github.com/bonniesjli/icm/blob/master/icm.py
    https://github.com/chagmgang/pytorch_ppo_rl/blob/master/model.py
    https://github.com/jcwleo/curiosity-driven-exploration-pytorch/blob/master/model.py
    https://zhuanlan.zhihu.com/p/66303476
    https://zhuanlan.zhihu.com/p/161948260
    https://github.com/uvipen/Street-fighter-A3C-ICM-pytorch/blob/4574cbfcbd148ed1d127ae053fe4afe943a18939/src/model.py
    https://github.com/adik993/ppo-pytorch/blob/master/curiosity/icm.py

    Notes:
    Not much useful. Large value will cause the trainning goes to wrong direction. small value will make no different but trainning much slower.
9. AlphaZero MSTC
   1.  https://github.com/suragnair/alpha-zero-general
   2.  https://github.com/louisnino/RLcode/tree/master/Alpha-Zero
   3.  https://github.com/plkmo/AlphaZero_Connect4
   4.  https://zhuanlan.zhihu.com/p/115867362
   5.  https://github.com/hijkzzz/alpha-zero-gomoku
   6.  https://github.com/junxiaosong/AlphaZero_Gomoku/
   7.  https://github.com/NeymarL/ChineseChess-AlphaZero
   8.  https://github.com/blanyal/alpha-zero
10. RunningMeanStd
    1.  https://github.com/jcwleo/curiosity-driven-exploration-pytorch/blob/e8448777325493dd86f2c4164e7188882fc268ea/train.py#L61