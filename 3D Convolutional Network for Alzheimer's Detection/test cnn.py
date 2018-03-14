import os
import tensorflow as tf
import tflearn
import numpy as np

data_dir='path_to_test_data'

datanp=[]                               #images
truenp=[]                               #labels

for file in os.listdir(data_dir):
    data=np.load(os.path.join(data_dir,file))
    datanp.append((data[0][0]))
    truenp.append(data[0][1])

sh=datanp.shape

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, sh[1], sh[2], sh[3], sh[4]])
net = tflearn.conv_3d(net, 16,5,strides=2,activation='leaky_relu', padding='VALID',weights_init='xavier',regularizer='L2',weight_decay=0.01)
net = tflearn.max_pool_3d(net, kernel_size = 3, strides=2, padding='VALID')
net = tflearn.conv_3d(net, 32,3,strides=2, padding='VALID',weights_init='xavier',regularizer='L2',weight_decay=0.01)
net = tflearn.normalization.batch_normalization(net)
net = tflearn.activations.leaky_relu (net)
net = tflearn.max_pool_3d(net, kernel_size = 2, strides=2, padding='VALID')
net = tflearn.dropout(net,0.5)
net = tflearn.fully_connected(net, 1024,weights_init='xavier',regularizer='L2')
net = tflearn.normalization.batch_normalization(net,gamma=1.1,beta=0.1)
net = tflearn.activations.leaky_relu (net)
net = tflearn.dropout(net,0.6)
net = tflearn.fully_connected(net, 512,weights_init='xavier',regularizer='L2')
net = tflearn.normalization.batch_normalization(net,gamma=1.2,beta=0.2)
net = tflearn.activations.leaky_relu (net)
net = tflearn.dropout(net,0.7)
net = tflearn.fully_connected(net, 128,weights_init='xavier',regularizer='L2')
net = tflearn.normalization.batch_normalization(net,gamma=1.4,beta=0.4)
net = tflearn.activations.leaky_relu (net)
net = tflearn.dropout(net,0.7)
net = tflearn.fully_connected(net, 3,weights_init='xavier',regularizer='L2')
net = tflearn.normalization.batch_normalization(net,gamma=1.3,beta=0.3)
net = tflearn.activations.softmax(net)
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
model = tflearn.DNN(net, checkpoint_path = 'drive/model/model.tfl.ckpt',max_checkpoints=3)                      #model definition

ckpt='path_to_latest_checkpoint'
model.load(ckpt)                                                                                                #loading checkpoints

model.evaluate(datanp,truenp)                                                                                   #evaluating the model, returns test accuracy
