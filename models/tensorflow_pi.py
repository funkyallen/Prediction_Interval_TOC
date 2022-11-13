import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Activation, BatchNormalization, Dense, Dropout
from tqdm import tqdm

from utils.tools import Loss_func, cal_pi_index

warnings.filterwarnings('ignore')


class PI_model(tf.keras.Model):
    def __init__(self, ini_bias_up, ini_bias_down):
        super(PI_model, self).__init__()
        self.f1 = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.2))
        self.f2 = Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.2))
        self.d1 = Dropout(0.2)
        self.f3 = Dense(2, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
                        bias_initializer=tf.keras.initializers.Constant(value=[ini_bias_up, ini_bias_down]))

    def call(self, inputs):
        x = self.f1(inputs)
        x = self.f2(x)
        x = self.d1(x)
        y = self.f3(x)
        return y


class History():
    def __init__(self, history):
        self.history = history


class TF_PI():
    def __init__(self, alpha=0.05, weight=[1, 0.5, 0.5]):
        self.soften_ = 160.
        self.alpha = alpha
        self.lamda = 1e-2
        self.model = None
        self.weight = weight

    def loss_func(self, y_true, y_pred):

        pi_upper = y_pred[:, 0]
        pi_lower = y_pred[:, 1]

        y_ture = tf.cast(y_true, dtype=tf.float32)
        num = tf.cast(tf.size(y_ture), tf.float32)  # sample size
        K_SU = tf.sigmoid(self.soften_ * (pi_upper - y_ture))
        K_SL = tf.sigmoid(self.soften_ * (y_ture - pi_lower))
        K_Soft = tf.multiply(K_SU, K_SL)

        pimw = tf.reduce_mean(tf.abs((pi_upper-pi_lower)/y_ture))
        piad = tf.reduce_mean(tf.abs(y_ture - (pi_upper+pi_lower)/2))
        picp_s = tf.reduce_mean(K_Soft)
        loss = self.weight[1]*pimw + self.weight[0]*self.lamda * num / \
            (self.alpha*(1-self.alpha)) * \
            tf.maximum(0., (1-self.alpha) - picp_s) + self.weight[2]*piad
        return loss

    def fit(self, train_x, train_y, val_data=None, batch_size=256, epochs=1000):
        if val_data is not None:
            val_x, val_y = val_data
            val_db = tf.data.Dataset.from_tensor_slices((val_x, val_y))
            val_db = val_db.batch(batch_size)

        ini_bias_up = 1.
        ini_bias_down = -1.
        self.model = PI_model(ini_bias_up, ini_bias_down)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-2, decay=1e-2)
        train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_db = train_db.shuffle(batch_size*5).batch(batch_size)

        picp = []
        pimw = []
        piad = []
        loss = []
        val_loss = []
        for epoch in tqdm(range(epochs)):
            for step, (x, y) in enumerate(train_db):

                with tf.GradientTape() as tape:
                    out = self.model(x)
                    train_loss = self.loss_func(y, out)

                grads = tape.gradient(
                    train_loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables))
            loss.append(train_loss)
            if val_data:
                for x, y in val_db:
                    val_out = self.model(x)
                    vali_loss = self.loss_func(y, val_out)
                val_loss.append(vali_loss)

        history = {"loss": loss, "val_loss": val_loss,
                   "picp": picp, "pimw": pimw, "piad": piad}
        history = History(history)
        return history

    def show_fit_process(self, history):

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        picp = history.history['picp']
        pimw = history.history['pimw']
        piad = history.history['piad']

        plt.figure(121)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss', fontsize=21)
        plt.legend(fontsize=18)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.tick_params(labelsize=16)

    def predict(self, test_x):
        result = self.model.predict(test_x)
        return result
