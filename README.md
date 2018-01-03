# Project VEO - Football detection and localization

## Project structure

```
root
├── getdata.py (download synthetic dataset)
├── provided_project (initial provided code)
│   ├── getdata.sh
│   ├── ballnoball.py
│   └── README.md
├── ball_noball
│   ├── checkpoints (saved models)
│   ├── summaries (tensorboard summaries)
│   ├── sample_validation (sample validation images)
│   ├── BatchLoader2.py
│   ├── BatchLoader2.py
│   ├── utils.py (utility file provided in lectures)
│   ├── model.py (script run on AWS for training the models)
│   ├── Network_testing.ipynb (Loading a checkpoint and running some sample validation images)
│   └── Network_testing.ipynb (Jupyter version of training script)
└── transfer_learning
    ├── data
    │    └── class.pbtxt
    ├── Frozen_state_rcnn
    │   ├── saved_model
    │   │   ├── variables
    │   │   └── saved_model.pb
    │   ├── checkpoint
    │   ├── frozen_inference_graph.pb
    │   ├── model.ckpt.data-00000-of-00001
    │   ├── model.ckpt.index
    │   └── model.ckpt.meta
    ├── models
    │   └── rcnn
    │   │   ├── train
    │   │   ├── model.ckpt.data-00000-of-00001
    │   │   ├── model.ckpt.index
    │   │   └── model.ckpt.meta
    ├── create_csv2.py
    ├── export_inference_graph.py
    ├── faster_rcnn_inception_v2_coco.config
    ├── generate_tfrecord.py
    └── train.py
```
## Dependencies

* Python 3.6
* Tensorflow 1.4
* numpy
* PIL

## How to replicate results (ball_noball)

1. Run getdata.py in root get download dataset
2. 

## How to replicate results (transfer learning)

1. What to do here?
2. 

## How to train the model and compare with provided results

1. Run getdata.py in root get download dataset
2. Run model.py in ball_noball folder
