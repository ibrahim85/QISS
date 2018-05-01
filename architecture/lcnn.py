import common
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import time

# Utility functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.4)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.2, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=(1, 1), padding='SAME'):
  return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                      padding=padding)

def max_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')

def convolutional_layers():
    x = tf.placeholder(tf.float32, [None, None, None])

    # First layer
    W_conv1 = weight_variable([5, 5, 1, 48])
    b_conv1 = bias_variable([48])
    x_expanded = tf.expand_dims(x, 3)
    h_conv1 = tf.nn.relu(conv2d(x_expanded, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    # Second layer
    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 1))

    # Third layer
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

    return x, h_pool3, [W_conv1, b_conv1,
                        W_conv2, b_conv2,
                        W_conv3, b_conv3]

def get_training_model():
    x, conv_layer, conv_vars = convolutional_layers()

    # Densely connected layer
    W_fc1 = weight_variable([32 * 8 * 128, 2048])
    b_fc1 = bias_variable([2048])

    conv_layer_flat = tf.reshape(conv_layer, [-1, 32 * 8 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)

    # Output layer
    W_fc2 = weight_variable([2048, 1 + 7 * common.OUTPUT_SHAPE[0]])
    b_fc2 = bias_variable([1 + 7 * common.OUTPUT_SHAPE[0]])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    return (x, y, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])

def get_train_model():
    x,y, params = get_training_model()

    inputs = tf.placeholder(tf.float32, [None, None, common.OUTPUT_SHAPE[0]])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell for forward and backward layer
    forwardH1 = rnn_cell.LSTMCell(common.num_hidden, use_peepholes=True, state_is_tuple=True)
    backwardH1 = rnn_cell.LSTMCell(common.num_hidden, use_peepholes=True, state_is_tuple=True)

    # The second output previous state and is ignored
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(forwardH1,backwardH1,x,seq_len,dtype=tf.float32)
    outputs=tf.concat(2,outputs)
    shape = tf.shape(inputs)

    batch_s, max_timesteps = shape[0], shape[1]
    weights = tf.Variable(tf.truncated_normal([common.num_hidden,
                                         common.num_classes],
                                        stddev=0.4), name="weights")
    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, 2*common.num_hidden])

    # Truncated normal with mean 0 and stdev=0.5
    W = tf.Variable(tf.truncated_normal([2*common.num_hidden, common.num_classes], stddev=0.5), name="W")

    # Zero initialization
    b = tf.zeros(shape=[common.num_classes],name='b')

    # Doing the affine projection
    logits = tf.matmul(outputs, W)+b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, common.num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    return logits, inputs, targets, seq_len, W, b
