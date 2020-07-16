import tensorflow as tf
import numpy as np

import logging
logger =tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

#Build model with Keras
# 1_Define layers
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
#l1 = tf.keras.layers.Dense(units=4)

# 2_Assemble layers into model

model = tf.keras.Sequential([l0])

# 3_compile model : give it loss and optimizer

model.compile(loss='mean_squared_error', optimizer= tf.keras.optimizers.Adam(0.1))

# 4_start_training

history = model.fit(celsius_q, fahrenheit_a, epochs = 100, verbose = 'False')

# 
print(history)
#print('\n',history.__dict__)
print('Training finished successfully ...')


# Plotting 

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.epoch, history.history['loss'])
#plt.show()
plt.savefig('Keras1/results/figure1.png')

# Predict using the model

f_p = model.predict([90.0])
f_true = 90.0*1.8 + 32

print("Pridected value = {} and true value = {}".format(f_p, f_true))

# Checking internal values of the layer

#print(l0.__dict__)
#print(dir(l0))
print(l0.get_weights())
print(l0._trainable_weights)