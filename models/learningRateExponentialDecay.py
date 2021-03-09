
import numpy as np
import tensorflow as tf

# class LearningRateExponentialDecay:
#     def __init__(self, initial_learing_rate, decay_epochs, decay_rate):
#         self.initial_learning_rate = initial_learing_rate
#         self.decay_epochs = decay_epochs
#         self.decay_rate = decay_rate
#
#     def __call__(self, epoch):
#         dtype = type(self.initial_learning_rate)
#         decay_epochs = np.array(self.decay_epochs).astype(dtype)
#         decay_rate = np.array(self.decay_rate).astype(dtype)
#         epoch = np.array(epoch).astype(dtype)
#         p = epoch / decay_epochs
#         lr = self.initial_learning_rate * np.power(decay_rate, p)
#         return lr

# class LearningRateExponentialDecay:
#     def __init__(self,initial_learning_rate,decay_epochs,decay_rate):
#         self.initial_learning_rate=initial_learning_rate
#         self.decay_epochs=decay_epochs
#         self.decay_rate=decay_rate
#     def __call__(self,epoch):
#         decay_epochs=np.array(self.decay_epochs).astype(tf.float32)
#         decay_rate=np.array(self.decay_rate).astype(tf.float32)
#         epoch = np.array(epoch).astype(tf.float32)
#         p = epoch/decay_epochs
#         lr = self.initial_learning_rate*np.power(decay_rate,p)
#         return lr
def LearningRateExponentialDecay(init_learningrate, decay_epochs, decay_rate, epoch):
    p = epoch / decay_epochs
    lr = init_learningrate * np.power(decay_rate, p)
    return lr