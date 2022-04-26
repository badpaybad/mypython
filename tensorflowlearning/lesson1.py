import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import model_from_yaml, model_from_json, Sequential
from keras.layers import Dense
import numpy

arr1=numpy.array( [[1,3],[3,1],[5,2]])

arr2=arr1.reshape(2,-1).astype("float32")/2

print(arr2.shape)
print(arr2)

#exit(0)

model = Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#model.compile(optimizer='sgd', loss='mean_squared_error')

#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

xs = numpy.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = numpy.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
''' 
the real: y = 3*x +1
'''
model.fit(xs, ys, epochs=500)

'''
x=10 => y = 3* 10 + 1 
predict expected around: 31 (+-1)
'''
print(model.predict([10.0]))
