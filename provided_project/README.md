
# Ball / No-ball - A toy example on real image data

Binary classification example with real images.

Install guide:

Install linux (Ubuntu 16.04)

Install anaconda (use python 3.6) - https://www.tensorflow.org/install/install_linux#InstallingAnaconda

conda create -n tensorflow

source activate tensorflow

(notice we change environment)

pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp36-cp36m-linux_x86_64.whl

pip install imageio

sudo apt-get install git

git clone http://github.com/recoord/ballnoball 

cd ballnoball

./getdata.sh

python ballnoball.py

Monitor using tensorboard from other terminal:

source activate tensorflow

(notice we change environment)

cd ballnoball

tensorboard --logdir tensorboard

browse to 127.0.0.1:6006


Jesper Taxbøl
jesper@sportcaster.dk
