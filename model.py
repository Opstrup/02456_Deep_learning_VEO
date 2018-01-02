#################################    Python runnable of jupyter notebook for AWS     #########################
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import utils 
from BatchLoader2 import BatchLoader2 as BatchLoader
from datetime import datetime

HEIGHT, WIDTH = 128, 128
NUMBER_CHANNELS = 1
NUMBER_OUTPUT_CLASS = 2
NUMBER_OUTPUT_LOC = 2
GPU_OPTS = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 150

#Build a building block for features, if poolsize and stride is the same, then the poolings are not overlapping
def conv_pool_drop(x,filters,kernel,pool,stride,keep_prob=1.0, counter=1, padding='same',use_avg=True,trace=False):
    with tf.variable_scope('conv'+str(counter)):
        lc1 = tf.layers.conv2d(x, filters, kernel, (1,1), padding=padding) #Do not downsample with cross-correlation
        if(trace is True):
            print(lc1)
    with tf.variable_scope('maxpool'+str(counter)):
        lp1 = tf.layers.max_pooling2d(lc1,pool,stride) #Pool it to downsample
        if(trace is True):
            print(lp1)
    if(use_avg is True):
        with tf.variable_scope('avgpool'+str(counter)): #We also introduce average pooling as maxpooling is rather agressive
            lap1 = tf.layers.average_pooling2d(lc1,pool,stride)
            if(trace is True):
                print(lap1)
        with tf.variable_scope('mergepool'+str(counter)):
            m1 = tf.concat([lp1,lap1],axis=3)
            if(trace is True):
                print(m1)
    else:
        m1 = lp1
    with tf.variable_scope('dropout'+str(counter)):
        ld1 = tf.layers.dropout(m1,rate=1.0-keep_prob)
        if(trace is True):
            print(ld1)
    return lc1,m1,ld1

def our_network(x_pl,hyper,trace=False,use_avg=True):
    #Our architecture
    x = x_pl
    #Architecture hyperparams
    blocknumbers = hyper[0] #3
    padding = hyper[1] #'same'
    filters = hyper[2] #[8,3,3]
    kernel_size = hyper[3] #[(7,7),(5,5),(3,3)]
    stride = hyper[4] #[(3,3),(3,3),(5,5)]
    pool_size = hyper[5] #[(5,5),(5,5),(3,3)]
    denseNum = hyper[6] #512
    denseNumReg = hyper[7] #512
    keep_prob = hyper[8]#tf.placeholder("float",name="KeepProbabilityPool")
    keep_prob2 = hyper[9]#tf.placeholder("float",name="KeepProbabilityDense")
    lc = []
    m = []
    ld = []
    #Need to initialize first layer manually, such that the loop can do it automatically later on
    lc1,m1,ld1 = conv_pool_drop(x,filters[0],kernel_size[0],pool_size[0],stride[0],keep_prob=keep_prob,counter=1,trace=trace,use_avg=use_avg)
    lc.append(lc1)
    m.append(m1)
    ld.append(ld1)
    for i in range(2,blocknumbers+1):
        i2 = min(i-1,len(filters)-1) ##If there are more layers, use the settings for the last one
        lc1,m1,ld1 = conv_pool_drop(ld[-1],filters[i2],kernel_size[i2],pool_size[i2],stride[i2],keep_prob=keep_prob,counter=i,trace=trace,use_avg=use_avg)
        lc.append(lc1)
        m.append(m1)
        ld.append(ld1)
    #Flatten and concatenate to evaluate the different features separately
    flattened = [] #Contains flattened feature maps on different levels
    with tf.variable_scope("Flatten"):
        for i in range(0,len(ld)):
            flattened.append(tf.layers.flatten(ld[i]))
            if(trace is True):
                print(flattened[i])
    with tf.variable_scope("Concatenate"):
        concat = tf.concat(flattened,axis=1)
        if(trace is True):
            print(concat)
    #Dense layers with dropout
    with tf.variable_scope("DenseInterpreterClass"):
        hiddenDense = tf.layers.dense(concat,denseNum,activation=tf.nn.relu,name="hidden_DenseClass")
        if(trace is True):
            print(hiddenDense)
        doDense = tf.layers.dropout(hiddenDense,rate=1.0-keep_prob2)
        if(trace is True):
            print(doDense)
    with tf.variable_scope("DenseInterpreterRegress"):
        hiddenDenseReg = tf.layers.dense(concat,denseNumReg,activation=tf.nn.relu,name="hidden_DenseRegress")
        if(trace is True):
            print(hiddenDenseReg)
        doDenseReg = tf.layers.dropout(hiddenDenseReg,rate=1.0-keep_prob2)
        if(trace is True):
            print(doDenseReg)
        
    #Output layers
    with tf.variable_scope("Predictions"):
        l_class = tf.layers.dense(doDense,NUMBER_OUTPUT_CLASS,activation=tf.nn.softmax,name="ClassificationGuess")
        if(trace is True):
            print(l_class)
        l_loc = tf.layers.dense(doDenseReg,NUMBER_OUTPUT_LOC,activation=tf.nn.relu,name="RegressionGuess")
        if(trace is True):
            print(l_loc)
    return l_class,l_loc

def build_networks(network_type,hyper,placeholders,trace=False):
    #Input type is common in all
    x_pl = placeholders[0]
    y_pl = placeholders[1]
    y_loc = placeholders[2]
    #Set up different network architecture
    if(network_type is 1):
        hyper.append(placeholders[3]) #Keep probability1
        hyper.append(placeholders[4]) #Keep probability2
        l_class, l_loc = our_network(x_pl,hyper,trace=trace,use_avg=True)
    elif (network_type is 2):
        # Do the same as before just without the averagepooling
        hyper.append(placeholders[3])
        hyper.append(placeholders[4])
        l_class, l_loc = our_network(x_pl,hyper,trace=trace,use_avg=False)
    else:
        #Network architecture provided by TAX
        l = x_pl
        with tf.variable_scope('conv2d'):
            l = tf.layers.conv2d(l, 16*4, [3, 3], 4, padding='VALID', name='conv1')
            if(trace is True):
                print(l)
            l = tf.layers.max_pooling2d(l, [2, 2], 2, name='pool1')
            if(trace is True):
                print(l)
            l = tf.layers.conv2d(l, 16*12, [3, 3], padding='VALID', name='conv2')
            if(trace is True):
                print(l)
            l = tf.layers.max_pooling2d(l, [2, 2], 2, name='pool2')
            if(trace is True):
                print(l)
            l = tf.layers.conv2d(l, 16*24, [3, 3], padding='VALID', name='conv3')
            if(trace is True):
                print(l)
            l = tf.layers.conv2d(l, 16*32, [3, 3], padding='VALID', name='conv4')
            if(trace is True):
                print(l)
        with tf.variable_scope('flatten'):
            l = tf.layers.flatten(l)
            if(trace is True):
                print(l)
        with tf.variable_scope('fully_connected'):
            l_class= tf.layers.dense(l,2, activation=tf.nn.softmax, name="fc_last")
            if(trace is True):
                print(l)
            #Added for regression; not present in provided architecture
            l_loc = tf.layers.dense(l,2, activation=tf.nn.relu, name="fc_loc")
    print('Model consits of ', utils.num_params(), 'trainable parameters.')
    
    #Loss, training, etc
    with tf.variable_scope('loss'):
        with tf.variable_scope('loss_class'):
            # computing cross entropy per sample for classification
            cross_entropy = -tf.reduce_sum(y_pl * tf.log(l_class+1e-8), reduction_indices=[1])

            # averaging over samples
            cross_entropy = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('cross_entropy',cross_entropy)

        with tf.variable_scope('loss_local'):
            #Get one-hot encoding for representing if the ball is present in the image
            ball_present = tf.cast(tf.equal(tf.argmax(y_pl,axis=1),tf.cast(1,tf.int64)),tf.float32)
            ball_present = tf.expand_dims(ball_present,1)
            #Compute mean squared error
            mse = tf.losses.mean_squared_error(y_loc,l_loc,weights=ball_present)#Ignore cases when ball is not present in image
            tf.summary.scalar('mean_sqared_error',mse)
        loss = cross_entropy+mse
        tf.summary.scalar('combined_loss',loss)
        reg_scale = 0.0005
        regularize = tf.contrib.layers.l2_regularizer(reg_scale)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        reg_term = sum([regularize(param) for param in params])
        loss += reg_term
        tf.summary.scalar('reg_combined_loss',loss)
    with tf.variable_scope('training'):
        # defining our optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        # applying the gradients
        train_op = optimizer.minimize(loss)


    with tf.variable_scope('performance_class'):
        # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
        correct_prediction = tf.equal(tf.argmax(l_class, axis=1), tf.argmax(y_pl, axis=1))

        # averaging the one-hot encoded vector
        accuracy_class = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('class_acc',accuracy_class)
    
    with tf.variable_scope('performance_local'):
        #Calculate eucledian distance between the predicted coordinates and the labelled ones
        ball_present = tf.cast(tf.equal(tf.argmax(y_pl,axis=1),tf.cast(1,tf.int64)),tf.float32)
        avgdist = ball_present*tf.norm((l_loc-y_loc)*128,axis=1,keep_dims=False,ord=2)#convert to pixel values
        avgdist = tf.reduce_sum(avgdist)/tf.maximum(1.0,tf.reduce_sum(ball_present)) #ignore predictions when ball is not present, safeguard if there is no ball in any of the pictures
        tf.summary.scalar('avg_pixel_dev',avgdist)
    merged_sum = tf.summary.merge_all()
    t = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    train_writer = tf.summary.FileWriter('./summaries/train/'+t,tf.get_default_graph())
    valid_writer = tf.summary.FileWriter('./summaries/valid/'+t,tf.get_default_graph())
    return l_class,train_op,accuracy_class,cross_entropy,l_loc,avgdist,loss,mse,merged_sum,train_writer,valid_writer

def experiment(hype=[3,'same',[8,3,3],[(7,7),(5,5),(3,3)],[(3,3),(3,3),(5,5)],[(5,5),(5,5),(3,3)],64,128], ntype=2, savename="default"):
    #Reset graph before building
    tf.reset_default_graph()
    #set up batch handler
    batchHandler = BatchLoader(dataset_folder_path="synthetic2_128x128", batch_size=BATCH_SIZE, keep_green=True, expand_dims=False)
    with tf.name_scope('input'):
        x_pl = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUMBER_CHANNELS], name='ImagePlaceholder')
    with tf.name_scope('output'):
        y_pl = tf.placeholder(tf.float64, [None, NUMBER_OUTPUT_CLASS], name='Classification')
        y_pl = tf.cast(y_pl, tf.float32)
        #We are predicting pixel locations, in percentage of the pixel width: 1.0,1.0 corresponds to 128,128
        y_loc = tf.placeholder(tf.float64, [None, NUMBER_OUTPUT_LOC], name='Regression')
        y_loc = tf.cast(y_loc, tf.float32)
    with tf.name_scope('keepProbs'):
        keep_prob = tf.placeholder("float",name="KeepProbabilityPool")
        keep_prob2 = tf.placeholder("float",name="KeepProbabilityDense")
    #Setting up placeholders and hyperparameters
    l_c,train_op,accuracy_class,cross_entropy,l_loc,avgdist,loss,mse,merged_sum,tw,vw = build_networks(ntype,hype,[x_pl,y_pl,y_loc,keep_prob,keep_prob2],trace=False)
    
    valid_loss_c,valid_loss_r,valid_loss, valid_accuracy_c,valid_accuracy_r = [], [],[],[],[]
    train_loss_c,train_loss_r,train_loss, train_accuracy_c,train_accuracy_r = [], [],[],[],[]
    saver = tf.train.Saver(max_to_keep=15,keep_checkpoint_every_n_hours=0.5) #keep all checkpoints
    validate_every_epoch = 1.0 ##Only validate after we trained on each image....
    nextvalid = 0.0+validate_every_epoch
    currentstep = 0 #keeping track of global steps
    currentvstep = 0
    with tf.Session(config=tf.ConfigProto(gpu_options=GPU_OPTS)) as sess:
        sess.run(tf.global_variables_initializer())
        print('Begin training loop')
        try:
            _train_loss_c, _train_accuracy_c = [], []
            _train_loss_r, _train_accuracy_r = [], []
            _train_loss = []
            while batchHandler.getEpoch() < MAX_EPOCHS: 
                currentstep = currentstep+1
                ## Run train op
                x_batch, y_batch, y_batch_loc = batchHandler.next_training_data()
                fetches_train = [train_op, cross_entropy, accuracy_class,avgdist,loss,mse,merged_sum]
                feed_dict_train = {x_pl: x_batch, y_pl: y_batch, y_loc:y_batch_loc, keep_prob: 0.8, keep_prob2: 0.6}
                _, _loss_c, _acc_c,_acc_r,_loss,_loss_r,summary = sess.run(fetches_train, feed_dict_train)
                _train_loss_c.append(_loss_c)
                _train_accuracy_c.append(_acc_c)
                _train_loss_r.append(_loss_r)
                _train_accuracy_r.append(_acc_r)
                _train_loss.append(_loss)
                #record training summary to directory
                tw.add_summary(summary,currentstep)
                ## It is time to validate
                if batchHandler.getEpoch() % 1.0 == 0 or batchHandler.getEpoch() > nextvalid:
                    print("Starting validation \n")
                    nextvalid += validate_every_epoch
                    #Record training statistics
                    train_loss.append(np.mean(_train_loss))
                    train_loss_r.append(np.mean(_train_loss_r))
                    train_loss_c.append(np.mean(_train_loss_c))
                    train_accuracy_c.append(np.mean(_train_accuracy_c))
                    train_accuracy_r.append(np.mean(_train_accuracy_r))

                    #Reset arrays until next validation
                    _train_loss_c, _train_accuracy_c = [], []
                    _train_loss_r, _train_accuracy_r = [], []
                    _train_loss = []
                    #Begin validation

                    fetches_valid = [cross_entropy, accuracy_class,mse,avgdist,loss,merged_sum]
                    _valid_loss, _valid_accuracy_c,_valid_loss_c,_valid_loss_r, _valid_accuracy_r = [], [],[],[],[]
                    while(batchHandler.getValid() != 1.0):
                        currentvstep = currentvstep+1
                        valid_x, valid_y, y_valid_loc = batchHandler.next_validation_data()
                        feed_dict_valid = {x_pl: valid_x, y_pl: valid_y, y_loc:y_valid_loc, keep_prob: 1.0, keep_prob2: 1.0}
                        _loss_c, _acc_c,_loss_r,_acc_r,_loss, summary = sess.run(fetches_valid, feed_dict_valid)
                        _valid_loss.append(_loss)
                        _valid_loss_c.append(_loss_c)
                        _valid_loss_r.append(_loss_r)
                        _valid_accuracy_c.append(_acc_c)
                        _valid_accuracy_r.append(_acc_r)
                        #Manually populate summary for validation
                        vw.add_summary(summary,currentvstep)
                    valid_loss.append(np.mean(_valid_loss))
                    valid_loss_c.append(np.mean(_valid_loss_c))
                    valid_loss_r.append(np.mean(_valid_loss_r))
                    valid_accuracy_c.append(np.mean(_valid_accuracy_c))
                    valid_accuracy_r.append(np.mean(_valid_accuracy_r))
                    summary = tf.Summary(value=[tf.Summary.Value(tag="loss_total_at_epoch", simple_value=valid_loss[-1]),tf.Summary.Value(tag="loss_class_at_epoch", simple_value=valid_loss_c[-1]),tf.Summary.Value(tag="loss_regress_at_epoch", simple_value=valid_loss_r[-1]),tf.Summary.Value(tag="class_acc_at_epoch", simple_value=valid_accuracy_c[-1]),tf.Summary.Value(tag="average_pixel_deviation", simple_value=valid_accuracy_r[-1]),
                        tf.Summary.Value(tag="train_loss_total_at_epoch", simple_value=train_loss[-1]),tf.Summary.Value(tag="train_loss_class_at_epoch", simple_value=train_loss_c[-1]),tf.Summary.Value(tag="train_loss_regress_at_epoch", simple_value=train_loss_r[-1]),tf.Summary.Value(tag="train_class_acc_at_epoch", simple_value=train_accuracy_c[-1]),tf.Summary.Value(tag="train_average_pixel_deviation", simple_value=train_accuracy_r[-1]),])
                    vw.add_summary(summary,batchHandler.getEpoch())
                    tw.flush()
                    vw.flush()
                    print("Epoch {} : Train Loss {:6.3f} (classification); {:6.3f} (regression); {:6.3f} (total), Train acc {:6.3f} (classification);{:6.3f} (regression),  Valid loss {:6.3f} (classification); {:6.3f} (regression); {:6.3f} (total),  Valid acc {:6.3f} (classification);{:6.3f} (regression)".format(
                            batchHandler.getEpoch(), train_loss_c[-1], train_loss_r[-1], train_loss[-1], train_accuracy_c[-1], train_accuracy_r[-1], valid_loss_c[-1], valid_loss_r[-1], valid_loss[-1], valid_accuracy_c[-1], valid_accuracy_r[-1]))
                    batchHandler.reset_validation()
                    save_path = saver.save(sess, "./checkpoints/final_"+savename+"_"+str(ntype)+"_"+str(batchHandler.getEpoch())+".ckpt")
            tw.close()
            vw.close()

        except KeyboardInterrupt:
            pass

def main():
    #Todo setup hyper parameter generation
    experiment(ntype=1,savename="avgpool")
    experiment(ntype=2,savename="default")
    experiment(ntype=3,savename="TAX")
if __name__ == '__main__':
    main()