import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import pandas as pd
import random
import os
import gzip
import sklearn
from sklearn.utils import shuffle

#initializations
home = '/home/ec2-user'
c_dir = '/home/ec2-user/MCIc_Segmented'
nc_dir = '/home/ec2-user/MCInc_Segmented'
width = 176
height = 256
depth = 256
nlabel = 2

#CREATING THE DATAFRAME WITH IMAGES*********************************************
#c is one class of patient, nc is another
c_dir = '/home/ec2-user/MCIc_Segmented'
nc_dir = '/home/ec2-user/MCInc_Segmented'

c_patients = os.listdir(c_dir)
nc_patients = os.listdir(nc_dir)

c_list = []
nc_list = []

#function to filter whether an image is to be trained on
def is_relevant(path):
    g_w = 'pve_0' in path
    unzipped = '.gz' not in path
    if g_w and unzipped:
        return True

#appending images to lists
for file in os.listdir(c_dir):
    if is_relevant(file):
        c_list.append(nib.load(c_dir + '/' + file))
for file in os.listdir(nc_dir):
    if is_relevant(file):
        nc_list.append(nib.load(nc_dir +  '/' + file))

#creating a datfaframe with all the images and their classifications, 0 corresponds to 'nc' and 1 corresponds to 'c'
names = ['Images', 'Classification']
data = c_list + nc_list
np.random.shuffle(data)
classifications = list(np.zeros(len(c_list), dtype = int)) + list(np.ones(len(nc_list), dtype = int))
df = pd.DataFrame()
df["Images"] = data
df["Classification"] = classifications
df = shuffle(df)

#function to return a tensor from a nibabel image
def make_tensor(image):
    array = image.get_data()
    data = tf.convert_to_tensor(array, dtype = 'int32')
    return(data)

#starting tensorflow session
sess = tf.InteractiveSession()

#CREATING THE NEURAL NETWORK****************************************************
#https://github.com/jibikbam/CNN-3D-images-Tensorflow/blob/master/simpleCNN_MRI.py#L65
#creating placeholders
#x = tf.placeholder(tf.float32, shape=[None, width*height*depth*1], name = 'images')
y_ = tf.placeholder(tf.float32, shape=[None, nlabel], name = 'classifications')

#tf.cast for changing int to float

#Defining a Variable with normally distibuted weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#Defining a bias variable with constant 0.1
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Convolution here: stride=1, zero-padded -> output size = input size
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

# Pooling: max pooling over 2x2 blocks
def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
  return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 5, 1, 32])  # shape of weight tensor = [5,5,1,32]
b_conv1 = bias_variable([32])  # bias vector for each output channel. = [32]

# Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
#x_image = tf.reshape(x, [-1,width,height,depth,1])
x_image = tf.placeholder(tf.float32, shape = [None, width, height, depth, 1])
print(x_image.get_shape)  #shape=(?, 166, 256, 256, 1) dtype=float32>>

# x_image * weight tensor + bias -> apply ReLU -> apply max-pool
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)  # conv2d, ReLU(x_image * weight + bias)
print(h_conv1.get_shape) #shape=(?, 166, 256, 256, 32)
h_pool1 = max_pool_2x2(h_conv1)  # apply max-pool
print(h_pool1.get_shape) #shape=(?, 42, 64, 64, 32)

## Second Convolutional Layer
# Conv then Max-pooling. 2nd layer will have 64 features for each 5x5 patch. (32 features -> 64 features)
W_conv2 = weight_variable([5, 5, 5, 32, 64]) # [5, 5, 32, 64]
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)  # conv2d, .ReLU(x_image * weight + bias)
print(h_conv2.get_shape) #shape=(?, 42, 64, 64, 64)
h_pool2 = max_pool_2x2(h_conv2)  # apply max-pool
print(h_pool2.get_shape)  #shape=(?, 11, 16, 16, 64)

## Densely Connected Layer (or fully-connected layer)
# fully-connected layer with 1024 neurons to process on the entire image
W_fc1 = weight_variable([16*16*11*64, 1024])  # [7*7*64, 1024]
b_fc1 = bias_variable([1024]) # [1024]]

h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*11*64])  # -> output image: [-1, 7*7*64] = 3136
print(h_pool2_flat.get_shape)  #shape=(?, 49152)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
print(h_fc1.get_shape) #shape=(?, 1024)

## Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop.get_shape)  #shape=(?, 1024)

## Readout Layer
W_fc2 = weight_variable([1024, nlabel]) # [1024, 10]
b_fc2 = bias_variable([nlabel]) # [10]

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(y_conv.get_shape)  #shape=(?, 2)

## Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

#TRAINING THE CNN
## Set up some crucial variables
N = len(df["Images"])
num_epochs = 10 # Total number of times we wish to cycle over the entire training dataset
batch_size = 1 # 2^0 - The batch size per iteration
num_iter_per_epoch = int(np.ceil(float(N) / batch_size)) # Total number of iterations per epoch

## Split up the data into training and test
prop_training = 0.9
split_point = int(np.floor(prop_training*N))
train_df = df.iloc[0:split_point]
test_df = df.iloc[split_point:]

# File open for logging
file = open('log.txt', 'w')

# Every 10 iterations, show the performance
show_every = 10

# Save the model to file
saver = tf.train.Saver()

## Main training loop
# For each epoch...
for i in range(num_epochs):
  # For each batch...
  for j in range(num_iter_per_epoch):
    s = "Training on batch {}".format(j + 1)
    print(s)
    file.write(s)
    # Make a dataframe that just contains our batch

    # Determines the end point of where we stop collecting
    # examples
    endpoint = min(N, (j + 1)*batch_size)

    # Collect the batch
    batch = train_df.iloc[j*batch_size : endpoint]

    # Convert the data into a NumPy array
    # This NumPy array will contain the voxels for the
    # batch
    data = np.zeros((batch_size, width, height, depth), dtype=np.float32)
    labels = np.zeros((batch_size, 2), dtype=np.float32)
    s = "Getting data for batch {}".format(j + 1)
    print(s)
    file.write(s)
    for k, (img, label) in enumerate(zip(batch["Images"], batch["Classification"])):
      # Create a padded image that would contain the voxel
      # data
      voxels = np.zeros((width, height, depth), dtype=np.float32)
      ary = img.get_data().astype(np.float32) # data is in (height, depth, width)
      #ary = np.transpose(ary, (1, 2, 0)) # Arrange data so it's (width, height, depth)
      (w, h, d) = ary.shape # Get the extent of the data

      # Place into zero-padded voxel space
      data[k, :w, :h, :d] = ary.copy()

      # Get the label as well
      lbl = int(label)
      labels[k, lbl] = 1.0

    # Update the graph with the data
    data = data[...,None] # Make this 5D
    s = "Updating graph for batch {}".format(j + 1)
    print(s)
    file.write(s)
    sess.run(train_step, feed_dict={x_image:data, y_:labels, keep_prob:0.7})

    # For every 10th iteration, let's show the performance
    # Show the loss and the accuracy of the training and test set
    if j % show_every == 0:
      s = "Evaluating test dataset..."
      print(s)
      file.write(s)
      num_batches = int(np.ceil(len(test_df) / batch_size))
      for k in range(num_batches):
        endpoint = min(len(test_df), (k + 1)*batch_size)
        btch = test_df.iloc[k*batch_size : endpoint]

        data = np.zeros((batch_size, width, height, depth), dtype=np.float32)
        labels = np.zeros((batch_size, 2), dtype=np.float32)
        total_loss = 0
        total_acc = 0
        for l, (img, label) in enumerate(zip(batch["Images"], batch["Classification"])):
          # Create a padded image that would contain the voxel
          # data
          voxels = np.zeros((width, height, depth), dtype=np.float32)
          ary = img.get_data().astype(np.float32) # data is in (height, depth, width)
          #ary = np.transpose(ary, (1, 2, 0)) # Arrange data so it's (width, height, depth)
          (w, h, d) = ary.shape # Get the extent of the data

          # Place into zero-padded voxel space
          data[l, :w, :h, :d] = ary.copy()

          # Get the label as well
          lbl = int(label)
          labels[l, lbl] = 1.0

        data = data[..., None]
        loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x_image:data, y_:labels, keep_prob:1.0})
        total_loss += loss
        total_acc += acc

      total_loss /= len(test_df)
      total_acc /= len(test_df)
      s = "Epoch {}, Iteration {}".format(i, j)
      print(s)
      file.write(s + "\n")
      s = "Loss on testing: {}.  Accuracy on testing: {}".format(total_loss, total_acc)
      print(s)
      file.write(s + "\n")
    file.write("\n")
    print("")

# Save the tensors to file for later
file.close()
tf.add_to_collection("y", y_)
tf.add_to_collection("x_image", x_image)
saver.save(sess, "models")
