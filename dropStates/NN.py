from sys import argv
from tensorflow.contrib.learn import DNNRegressor
from sklearn import model_selection, metrics
import tensorflow
import numpy

states = open(argv[1], 'r')

data = []
labels = []

for line in states:
    score, vector = line.split('|||')
    labels.append(float(score.strip()))
    data.append([float(x.strip()) for x in vector.strip().split(' ')])

states.close()
labels = numpy.array(labels).astype('float32')
data = numpy.array(data).astype('float32')
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.2, random_state=42)

feature_columns = tensorflow.contrib.learn.infer_real_valued_columns_from_input(x_train)
regressor = DNNRegressor(feature_columns=feature_columns, hidden_units=[512, 512])
regressor.fit(x_train, y_train, steps=5000, batch_size=10)

y_predicted = list(regressor.predict(x_test))
score = metrics.mean_squared_error(y_predicted, y_test)
