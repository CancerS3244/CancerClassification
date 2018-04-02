# imports
import glob, os, json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# imaging library
# run `pip install Pillow` if ModuleNotFoundError
from PIL import Image 
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

def build_dataset():
    img_filenames = []
    labels = []
    counter = 0

    D = []
    for fname in glob.glob("ISIC-images/*/*.jpg", recursive=True):
        (img_id, ext) = os.path.splitext(fname)
        with open(os.path.join(img_id)+".json") as f:    
            metadata = json.load(f)
            lab = metadata["meta"]["clinical"]["benign_malignant"]
        
        # resize image
        im = Image.open(fname)
        im = im.resize((28,28))
        
        # cheat to ensure classes are balanced
        if counter < 50 and lab!="malignant":
            continue
        if 50<counter<100 and lab!="benign":
            continue

        img_filenames.append(fname)
        labels.append(1 if lab=="malignant" else 0)

        im = np.asarray(im)
        im = np.reshape(im,(28*28,3))
        D.append(im)

        # read only 300 images
        counter +=1
        if counter == 300: break
    return (np.asarray(D,dtype=np.float32), np.asarray(labels, dtype=np.int32))

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer with shape [batch_size, image_width, image_height, channels]
    # -1 for batch size ==> dynamically computed based on input values
    # 28,28 for img width and height
    # 1 channel (monochrome)
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 3])

    # Convolutional Layer #1
    # Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    # Pooling Layer #1
    # Performs max pooling with a 2x2 filter and stride of 2
    # pool regions do not overlap
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    # Applies 64 5x5 filters, with ReLU activation function
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
    # conv2 shape: [batchsize, 14, 14, 64]
    # max pooling with a 2x2 filter and stride of 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer (same as fully connected)
    # pool2 width and pool2 height = 7
    # pool2 channels = 64
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # 1024 units, ReLU activation
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # dropout layer has shape [batch_size, 1024]
    # dropout = tf.layers.dropout(
    #         inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    # 2 units, one for each digit target class (benign or cancerous).
    logits = tf.layers.dense(inputs=dense, units=2)

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

def main():
    # Load training and eval data
    D, labels = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(D, labels)
    

    # actual training of model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train}, # shape (N, w*d)
        y=y_train, # shape
        batch_size=10,
        num_epochs=2, # num of epochs to iterate over data. If `None` will run forever.
        shuffle=True
        )
    cs3244_classifier.train(
        input_fn=train_input_fn,
        steps=None, # train until input_fn stops
        hooks=[logging_hook]
        )
    
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=y_test,
        num_epochs=1,
        shuffle=False
        )
    eval_results = cs3244_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
# Create the Estimator
cs3244_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, 
    model_dir="/tmp/mnist_convnet_model"
    )

# Set up logging for predictions
# uncomment to log probabilities
# tensors_to_log = {"probabilities": "softmax_tensor"}
tensors_to_log = {}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, 
    every_n_iter=50
    )

main()