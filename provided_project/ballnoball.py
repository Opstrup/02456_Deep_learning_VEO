from __future__ import division
from __future__ import print_function

import os
import io
import sys
import math
import glob
import time
import queue
import random
import shutil
import imageio
import zipfile
import numpy as np
from subprocess import check_output
from PIL import Image, ImageDraw
import scipy.ndimage
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import tensorflow.contrib.losses as losses
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
import tensorflow.contrib.slim as slim
from threading import Thread
from itertools import product
from tensorflow.python.framework import graph_util

SOURCE_WIDTH = 128


def batch_iterator(zip, files, batch_size):    
    batches = math.floor(len(files)/batch_size)
    
    #tell how many batches there is data for
    yield(batches)

    file_queue = queue.Queue(maxsize=0)
    random.shuffle(files)
    
    #move all names into name queue so workers can get started
    for n in files:
        file_queue.put(n)
    
    #create queue for workers to preprocess image data 
    #into batches ready to be loaded directly
    data_queue = queue.Queue(maxsize=64)

    global running
    #preprocess images and pack them in batches
    def preprocess(file_q, data_q, batch_size, thread_id):
        #global running
        while running:
            input_batch = []
            target_batch = []
            source_image_paths = []
            
            for i in range(batch_size):
                #get next filename
                image_path_in_zip = file_q.get()

                #Load image from archive
                img = Image.open(io.BytesIO(zip.read(image_path_in_zip)))
                
                #This dataset store label in filename
                target = [1.0,0.0]
                if "/noball/" in image_path_in_zip:
                    target = [0.0,1.0]
                    
                #cast to numpy and add to batch - all data as uint8
                input_batch.append(np.asarray(img).astype(np.uint8))
                target_batch.append(np.array(target).astype(np.uint8))
                file_q.task_done()
                
                #reinsert filename so cycling can happen
                file_q.put(image_path_in_zip)

            #put batch in list
            data_q.put((np.stack(input_batch), np.stack(target_batch))) 
        #print("Ending thread ", thread_id)
    
    
    #kickoff some preprocessing threads
    workers = []
    for i in range(4):
        worker = Thread(target=preprocess, args=(file_queue, data_queue, batch_size, i))
        worker.setDaemon(True)
        worker.start()
        workers.append(worker)

    while running:
        inputs, targets = data_queue.get()
        data_queue.task_done()
        yield (inputs, targets)
    while not data_queue.empty():
        data_queue.get()
        #print("purging data")
    
    #print("Ending batchloader ", zip_file)


def experiment(c):    
    global running
    running = True
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_name = "%s_lr%f_b%d" % (t, c['learning_rate'], c['batch_size'])
    
    print("\n\n\n***********************")
    print("* Starting experiment: %s" % (experiment_name))
    print("***********************")
    
    tf.reset_default_graph();

    
    #control randomization
    tf.set_random_seed(c['random_seed'])
    random.seed = c['random_seed']
    np.random.seed(c['random_seed'])
    
    print(">>>>>>>>>> Building graph")

    #Begin building graph
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
    increment_global_step_op = tf.assign(global_step, global_step+1)

    #Initializer needed for weights and bios
    xi = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32, seed=c['random_seed'])

    train_summaries = []
    valid_summaries = []

    #Input image 128x128 in range 0-255 -> converted to range -1..1
    with tf.name_scope("input"):
        x_uint8 = tf.placeholder(tf.uint8, shape=(None, SOURCE_WIDTH, SOURCE_WIDTH), name="x")
        x = tf.reshape(tf.to_float(x_uint8), (-1, SOURCE_WIDTH, SOURCE_WIDTH,1))
        x = (x/255.0 - 0.5) * 2.0
        
    #Target image 1x2 in range 0-255 -> converted to range 0..1
    with tf.name_scope("target"):
        y = tf.placeholder(tf.float32, shape=(None, 2), name="labels")
        
        
    l = x
    with tf.variable_scope('conv2d'):
        l = slim.conv2d(l, 16*4, [3, 3], 4, padding='VALID', scope='conv1')
        print(l)
        l = slim.max_pool2d(l, [2, 2], 2, scope='pool1')
        print(l)
        l = slim.conv2d(l, 16*12, [3, 3], padding='VALID', scope='conv2')
        print(l)
        l = slim.max_pool2d(l, [2, 2], 2, scope='pool2')
        print(l)
        l = slim.conv2d(l, 16*24, [3, 3], padding='VALID', scope='conv3')
        print(l)
        l = slim.conv2d(l, 16*32, [3, 3], padding='VALID', scope='conv4')
        print(l)

    l = slim.flatten(l)
    with tf.variable_scope('fully_connected'):
        l = slim.fully_connected(l, 2, activation_fn=None, scope="fc_last", weights_initializer=xi)
        

    response = l
    
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=response))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(response, 1)), tf.float32))
        
        train_summaries.append(tf.summary.scalar("training_loss", loss))
        train_summaries.append(tf.summary.scalar("training_accuracy", accuracy))
        valid_summaries.append(tf.summary.scalar("validation_loss", loss))
        valid_summaries.append(tf.summary.scalar("validation_accuracy", accuracy))




    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(c['learning_rate']).minimize(loss)
    

    response = softmax(response)


    with tf.name_scope('training'):
        SAMPLES_IN_IMAGE = min(c['batch_size'], 8)
        response_img = tf.reshape(response, (-1, 2, 1, 1))
        response_img = tf.image.resize_images(response_img, [128,64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        target_img = tf.image.resize_images(tf.reshape(y, (-1, 2, 1, 1)), [128,64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        full_img = tf.concat([x*0.5+0.5,target_img, response_img], 2)
        full_img = tf.cast(full_img*255, tf.uint8)
        full_img = tf.reshape(full_img, (-1, 128 * SAMPLES_IN_IMAGE, 128*2,1))

        train_summaries.append(tf.summary.image('full_img_train', full_img, 1))
        valid_summaries.append(tf.summary.image('full_img_valid', full_img, 1))

    train_summary_op = tf.summary.merge(train_summaries)
    valid_summary_op = tf.summary.merge(valid_summaries)    
    


    session = tf.Session()        
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    train_data = batch_iterator(c['dataset_zip'], list(c['training_filelist']), c['batch_size'])
    batches_trainingset = next(train_data)
    validation_data = batch_iterator(c['dataset_zip'], list(c['validation_filelist']), c['batch_size'])
    batches_validationset = next(validation_data)

    iterations = math.floor(batches_trainingset*c['epochs'])

    writer = tf.summary.FileWriter("tensorboard/" + experiment_name)
    writer.add_graph(session.graph)

    train_loss = 0.0
    valid_loss = 0.0
    train_accuracy = 0.0
    valid_accuracy = 0.0

    print(">>>>>>>>>> Kickoff training.")
    start_time = time.time()
    last_validation = start_time
    for i in range(int(iterations)):
        #Fetch a batch
        inputs, targets = next(train_data)
        _, train_loss, train_accuracy, s, global_step_val = session.run(
            fetches=[train_step, loss, accuracy, train_summary_op, increment_global_step_op], 
            feed_dict={x_uint8: inputs, y: targets})
        writer.add_summary(s, global_step_val)
        

        #validate once in a while
        if  i%10 == 9:
            inputs, targets = next(validation_data)
            [valid_loss, valid_accuracy, s] = session.run(
                fetches=[loss, accuracy, valid_summary_op], 
                feed_dict={x_uint8: inputs, y: targets})
            writer.add_summary(s, global_step_val)
            
        #print status
        outstr = '\rE: %s    ' % (experiment_name)
        outstr += 'Batch: %07d/%07d/%0.7d    ' % (i+1,iterations,global_step_val)
        outstr += 'Train: %f|%f   ' % (train_loss, train_accuracy)
        outstr += 'Validation: %f|%f    ' % (valid_loss, valid_accuracy)
        now = time.time()
        if(now > start_time and i > 0):
            speed = c['batch_size']*i*1.0/(now - start_time)
            remaining = ((iterations - i - 1)*c['batch_size'])/speed
            outstr += 'Speed: %0.1f    Seconds left: %0.1f   ' % (speed, remaining)
            
        sys.stdout.write(outstr)
        sys.stdout.flush()
    
    session.close()
    
    print(">>>>>>>>>> Stopping batch loaders gracefully")
    running = False
    try:
        while True:
            next(train_data)        
    except StopIteration:
        pass
    try:
        while True:
            next(validation_data)        
    except StopIteration:
        pass
    print(">>>>>>>>>> Experiment %s done" % (experiment_name))


#run range of experiments
def main():
    print("Tensorflow version: ", tf.__version__)

    dataset_zip = 'ballnoball2.zip'
    
    print("Opening dataset archive:", dataset_zip)
    training_files = []
    validation_files = []
    dataset_zip=zipfile.ZipFile(dataset_zip, mode='r')
    for n in dataset_zip.namelist():
        if n.lower().endswith(".png"):
            if "training" in n:
                training_files.append(n)
            else:
                validation_files.append(n)
    
    random.seed = 42
    random.shuffle(training_files)
    random.shuffle(validation_files)
    print("%d training images"%(len(training_files)))
    print("%d validation images"%(len(validation_files)))
    
    configurations = []
    for lr in [x*0.0007 + 0.0001 for x in range(10)]:
        configurations.append({
                'random_seed': 42,
                'dataset_zip' : dataset_zip,
                'training_filelist' : training_files,
                'validation_filelist' : validation_files,
                'learning_rate' : lr,
                'batch_size' : 16,
                'epochs': 1.0,
            })
    for c in configurations:
        experiment(c)
    print("fin")

if __name__ == '__main__':
    main()


