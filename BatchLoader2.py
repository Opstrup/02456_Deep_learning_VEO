import math
import io
import os
import numpy as np
from queue import Queue
import random
from threading import Thread
from PIL import Image, ImageDraw

class BatchLoader2:
    
    def __init__(self, dataset_folder_path, batch_size, keep_green, expand_dims):
        global running
        running = True
        # Initialize variables
        self.epochs_complete = 0
        self.valid_complete = 0
        self.batchCounter = 0
        self.dataset_folder_path = dataset_folder_path
        self.batch_size = batch_size

        data = self.gather_images(dataset_folder_path+"/training")
        self.dataLength = len(data)
        
        data_v = self.gather_images(dataset_folder_path+"/validation")
        self.validLength = len(data_v)
        random.shuffle(data)
        random.shuffle(data_v)
        self.train_data = self.prepare_data(data,batch_size)
        self.valid_data = self.prepare_data(data_v,batch_size)
        self.keep_g = keep_green
        self.exp_dim = expand_dims

    def gather_images(self, image_folder_path):
        images_file_path = []
        for root, dirs, files in os.walk(image_folder_path):
            for file in files:
                images_file_path.append(os.path.join(root, file))

        return images_file_path
        
    def prepare_data(self, data, batchSize):
        #create infinit queue for multi threads
        file_queue = Queue(maxsize=0)
        #move all names into name queue so workers can get started
        for n in data:
            file_queue.put(n)
        
        #create queue for workers to preprocess image data 
        #into batches ready to be loaded directly
        data_queue = Queue(maxsize=64)
        #preprocess images and pack them in batches
        def preprocess(file_q, data_q, batch_size, thread_id):

            while running:
                input_batch = []
                target_batch = []
                target_batch2 = []
                source_image_paths = []

                for i in range(batch_size):
                    #get next file
                    image_path = file_q.get()

                    #Load image PIL.PngImagePlugin.PngImageFile
                    img = Image.open(image_path)
                    #Normalize to [0-1]
                    img = np.asarray(img).astype(np.float32)/255.0
                    if(self.exp_dim is True):
                        img = np.expand_dims(img, axis=2)
                    if(self.keep_g is True):
                        #Remove unnecessary channels
                        img = np.delete(img,[0,2],2)
                    file_name = os.path.basename(image_path).split('_')
                    x = float(file_name[4])/128.0
                    y = float(file_name[5])/128.0
                    ballNoBall = file_name[6]
                    #Generate label value (label stored in filename)
                    if(ballNoBall == "1.png"):
                        target = [0,1]
                        target2 = [x,y]
                    else:
                        target = [1,0]
                        target2 = [-1,-1] #The second argument is a placeholder, as the ball is not present in the image
                    #cast to numpy and add to batch - all data as uint8
                    input_batch.append(np.asarray(img).astype(np.float32))
                    target_batch.append(np.array(target).astype(np.float32))
                    target_batch2.append(np.array(target2).astype(np.float32))
                    file_q.task_done()

                    #reinsert filename so cycling can happen
                    #file_q.put(image_path)

                #put batch in list
                data_q.put((np.stack(input_batch), np.stack(target_batch), np.stack(target_batch2))) 
    
        #kickoff some preprocessing threads
        workers = []
        for i in range(1):
            worker = Thread(target=preprocess, args=(file_queue, data_queue, batchSize, i))
            worker.setDaemon(True)
            worker.start()
            workers.append(worker)
        
        while running:
            inputs, targets, t2 = data_queue.get()
            data_queue.task_done()
            yield (inputs, targets, t2)
        while not data_queue.empty():
            data_queue.get()
        
    def next_training_data(self):
        temp = next(self.train_data)
        batches = max(1,math.floor(float(self.dataLength)/self.batch_size))
        self.batchCounter += 1
        if(self.batchCounter % batches == 0):
            self.epochs_complete += 1
            self.batchCounter = 0
            self.epoch_complete()
        return temp
    
    def next_validation_data(self):
        self.valid_complete += 1
        temp = next(self.valid_data)
        return temp

    def epoch_complete(self):
        #running = False
        data = self.gather_images(self.dataset_folder_path+"/training")
        random.shuffle(data)
        #running = True
        self.train_data = self.prepare_data(data,self.batch_size)

    def reset_training_data(self):
        self.epochs_complete = 0
        self.batchCounter = 0
        self.epoch_complete()

    def reset_validation(self):
        running = False
        self.valid_complete = 0
        data_v = self.gather_images(self.dataset_folder_path+"/validation")
        self.validLength = len(data_v)
        random.shuffle(data_v)
        running = True
        self.valid_data = self.prepare_data(data_v,self.batch_size)

    def getEpoch(self):
        batches = max(1,math.floor(float(self.dataLength)/self.batch_size))
        return self.epochs_complete + float(self.batchCounter)/batches
    def getValid(self):
        batches = max(1,math.floor(float(self.validLength)/self.batch_size))
        return float(self.valid_complete)/batches