# imports
import glob, os, json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

# imaging library
# run `pip install Pillow` if ModuleNotFoundError
from PIL import Image 
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)

MAX_NUMBER_OF_DATA = 450

def build_dataset():
  number_of_data = 0

  img_file_names = []
  labels = []
  data = []
  benign_labels = []
  malign_labels = []
  benign_data = []
  malign_data = []
  benign_num = 0
  malign_num = 0

  for image_file_path in glob.glob("ISIC-data/resized_images_width_64/*.jpg", recursive=True):
    (image_name, ext) =  os.path.splitext(os.path.basename(image_file_path))
    meta_data_path = os.path.join('ISIC-data', 'Descriptions', image_name)

    if (os.path.isfile(meta_data_path)):
      with open(meta_data_path) as meta: 
        metadata = json.load(meta)
        label = metadata["meta"]["clinical"]["benign_malignant"]
        img = Image.open(image_file_path)
        img = img.resize((56,56))

        if (label == "benign"):
          if (benign_num < MAX_NUMBER_OF_DATA):
            img = np.asarray(img)
            img = np.reshape(img, (56 * 56, 3))
            benign_data.append(img)
            benign_labels.append(0)
            benign_num = benign_num + 1
            
          elif (benign_num == MAX_NUMBER_OF_DATA):
            if (malign_num < MAX_NUMBER_OF_DATA):
              continue
            else:
              break            
        elif (label == "malignant"):
          if (malign_num < MAX_NUMBER_OF_DATA):
            img = np.asarray(img)
            img = np.reshape(img, (56 * 56, 3))
            malign_data.append(img)
            malign_labels.append(1)
            malign_num = malign_num + 1;
            
          elif (malign_num == MAX_NUMBER_OF_DATA):
            if (benign_num < MAX_NUMBER_OF_DATA):
              continue
            else:
              break 
  
  # returns 2-tuple of data and labels.
  for i in range(MAX_NUMBER_OF_DATA):
    data.append(benign_data[i])
    data.append(malign_data[i])
    labels.append(benign_labels[i])
    labels.append(malign_labels[i])

  return (np.asarray(data, dtype=np.float32), np.asarray(labels, dtype=np.int32)) 

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer with shape [batch_size, image_width, image_height, channels]
  # -1 for batch size ==> dynamically computed based on input values
  # resize width and height to 56 by 56
  # 3 channels since image has RGB channels
  input_layer = tf.reshape(features["x"], [-1,56,56,3])
  
  # Convolutional Layer #1
  # Applies 96 11x11 filters (extracting 11x11-pixel subregions), with ReLU activation function
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=96,
      kernel_size=[11, 11],
      strides=4,
      padding="same",
      activation=tf.nn.relu)
  # Pooling Layer #1
  # Performs max pooling with a 2x2 filter and stride of 2
  # pool regions do not overlap
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)
  print(pool1)

  # Convolutional Layer #2 and Pooling Layer #2
  # Applies 64 5x5 filters, with ReLU activation function
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=[11, 11],
      padding="same",
      activation=tf.nn.relu)
  # conv2 shape: [batchsize, 14, 14, 64]
  # max pooling with a 2x2 filter and stride of 2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)
  print(pool2)

  # Convolutional Layer #2 and Pooling Layer #2
  # Applies 64 5x5 filters, with ReLU activation function
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=384,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Outputs [BATCH_SIZE , 11, 11, 256]
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=1)
  print(pool5)
  
  # Flatten for neural net
  pool5_flat = tf.reshape(pool5, [-1, 11 * 11 * 256])

  
  dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
  dense2 = tf.layers.dense(inputs=dense1, units=4096, activation=tf.nn.relu)

  dropout2 = tf.layers.dropout(
    inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
  )

  logits = tf.layers.dense(inputs=dropout2, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Create the Estimator
cs3244_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="alexmodels/")

# Set up logging for predictions
# uncomment to log probabilities
# tensors_to_log = {"probabilities": "softmax_tensor"}
tensors_to_log = {}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def main():
  NUM_TRAINING_ITERATIONS = 3
  max_value = 0.0
  
  # Load training and eval data
  data, labels = build_dataset()
  training_data, test_data, training_labels, test_labels = train_test_split(
      data, labels, test_size=0.33, shuffle=False)

  # actual training of model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": training_data}, # shape (N, w*d)
    y=training_labels, # shape
    batch_size=10,
    num_epochs=5, # num of epochs to iterate over data. If `None` will run forever.
    shuffle=True)
  
  for i in range(0, NUM_TRAINING_ITERATIONS):
    cs3244_classifier.train(
      input_fn=train_input_fn,
      steps=100000 # train until input_fn stops
    )
  
  # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=True)
    eval_results = cs3244_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    max_value = max(max_value, eval_results['accuracy'])
  print("max acc: {}".format(max_value))


def debug_print():
  temp_x, temp_y = build_dataset()
  input_layer = tf.reshape(temp_x[0], [-1,56,56,3])
  # Convolutional Layer #1
  # Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=96,
      kernel_size=[11, 11],
      strides=4,
      padding="same",
      activation=tf.nn.relu)
  # Pooling Layer #1
  # Performs max pooling with a 2x2 filter and stride of 2
  # pool regions do not overlap
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)
  print(pool1)

  # Convolutional Layer #2 and Pooling Layer #2
  # Applies 64 5x5 filters, with ReLU activation function
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=[11, 11],
      padding="same",
      activation=tf.nn.relu)
  # conv2 shape: [batchsize, 14, 14, 64]
  # max pooling with a 2x2 filter and stride of 2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)
  print(pool2)

  # Convolutional Layer #2 and Pooling Layer #2
  # Applies 64 5x5 filters, with ReLU activation function
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=384,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Outputs [BATCH_SIZE , 11, 11, 256]
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=1)
  print(pool5)
  
  # Flatten for neural net
  pool5_flat = tf.reshape(pool5, [-1, 11 * 11 * 256])


  dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
  dense2 = tf.layers.dense(inputs=dense1, units=4096, activation=tf.nn.relu)

  dropout2 = tf.layers.dropout(
    inputs=dense2, rate=0.4, training=False#mode == tf.estimator.ModeKeys.TRAIN
  )

  logits = tf.layers.dense(inputs=dropout2, units=2)


main()
#debug_print()