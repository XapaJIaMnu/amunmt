import tensorflow as tf
import numpy
from sklearn.model_selection import train_test_split
from tensorflow.python.client import timeline

def init_weight(shape, name):
    """Initialize a weight matrix"""
    weight = tf.random_normal(shape, dtype='float32', name=name, stddev=0.1)
    return tf.Variable(weight)

def forwardpass(X, X_ID, w_1, b_1):
    """Forward pass of the NN"""
    multiplication = tf.add(tf.matmul(X, w_1), b_1)
    return tf.gather(tf.transpose(multiplication), X_ID)
    #return tf.gather_nd(multiplication, [X_ID, 1])

def FFNN_train(data):
    """Defines the feed forward neural network"""
    x_size = 500 # Size of the hidden layer input
    y_size = 90321 # Vocab size

    # tf Graph Input
    X = tf.placeholder("float", name="X", shape=[1, x_size])
    X_ID = tf.placeholder("int32", name="X_ID", shape=[1])
    Y = tf.placeholder("float", name="Y", shape=[1])

    # init weights
    w_1 = init_weight((x_size, y_size), 'w1')
    b_1 = init_weight((1, y_size), 'b1')

    # Forward pass
    y_hat = forwardpass(X, X_ID, w_1, b_1)

    cost = tf.pow((Y - y_hat), 2)

    #Use adam to optimize and initialize the cost
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
    init_op = tf.global_variables_initializer()

    x_vec_train, x_vec_test, x_id_train, x_id_test, y_train, y_test = train_test_split(data[0], data[1], data[2], test_size=0.2, random_state=42)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        sess.run(init_op)

        # Fit all training data
        for epoch in range(10):
            for (x_vec, x_id, y) in zip(x_vec_train, x_id_train, y_train):
                sess.run(train_op, feed_dict={X: x_vec.reshape(1,500), X_ID: x_id.reshape(1), Y: y.reshape(1)})#, options=run_options, run_metadata=run_metadata)

            # Create the Timeline object, and write it to a json
            #tl = timeline.Timeline(run_metadata.step_stats)
            #ctf = tl.generate_chrome_trace_format()
            #with open('timeline' + str(epoch) + '.json', 'w') as f:
            #    f.write(ctf)

            # Display logs per epoch step
            c = sess.run(cost, feed_dict={X: x_vec.reshape(1,500), X_ID: x_id.reshape(1), Y: y.reshape(1)})
            print("Epoch:" + str(epoch + 1) + " cost= " + str(c))

def preprocess(filename):
    """Creates the dataset"""
    train_file = open(filename, 'r')
    X_vec = []
    X_wID = []
    Y = []
    for line in train_file:
        if line.strip() == "":
            continue
        score, wordID, vec = line.strip().split(' ||| ')
        Y.append(score)
        X_wID.append(wordID)
        X_vec.append([float(x) for x in vec.strip().split(' ')])
    train_file.close()
    return (numpy.array(X_vec).astype('float32'), numpy.array(X_wID).astype("int32"), numpy.array(Y).astype('float32'))
