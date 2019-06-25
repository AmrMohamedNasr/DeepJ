#!/bin/bash

ls
ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
python3 -m pip install gputil
python3 -m pip install psutil
python3 -m pip install humanize
python3 -m pip install PyDrive
python3 -m pip install joblib
unzip data.zip

git clone https://github.com/louisabraham/python3-midi.git
sudo apt-get install -y swig
cd python3-midi
sudo python3 setup.py install
cd ..

python3 -m pip install keras==2.0.8

python3 train.py
