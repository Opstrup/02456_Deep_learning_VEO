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
│   ├── utils.py (utility file provided in lectures)
│   ├── model.py (script run on AWS for training the models)
│   ├── Network_testing.ipynb (Loading a checkpoint and running some sample validation images)
│   └── Network_training.ipynb (Jupyter version of training script)
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
(Training:)
2. Run Network_training.ipynb
(Checking validation images:)
3. Run Network_testing.pynb

Summaries are located in ball_noball/summaries

## How to replicate results (transfer learning)

1. Follow the step from  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
   to install the needed dependencies for Tensorflow object detection api.
2. Use create_csv2.py to create two csv files. One for the traning images and one for testing.
3. Use generate_tfrecord.py to make TFrecords from the csv and images.
4. Select the wanted model from models in transfer learning
5. Modify the config file that match the model, such that: fine_tune_checkpoint point to the model checkpoint. input_path in train_input_reader point to train TFrecord
   and input_path in eval_input_reader points to test TFrecord. All label_map_path point to class.pbtxt.
6. Run train.py by telling it where training directory and config file is.
7. When sufficient trained use export_inference_graph.py to extract a frozen state from the training checkpoints.
8. Use either custom_rcnn_object_detection.ipynb or custom_ssd_object_detection.ipynb to run the created frozen state. (Remember to modify the location of the model and images in the ipynb).


## How to train the model and compare with provided results

1. Run getdata.py in root get download dataset
2. Run model.py in ball_noball folder
