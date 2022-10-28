import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from tqdm import tqdm
import warnings
from utils.tools import Piei, cal_pi_index
import numpy as np

warnings.filterwarnings('ignore')


class PI_model(tf.keras.Model):
    def __init__(self, ini_bias_up, ini_bias_down):
        super(PI_model, self).__init__()
        self.f1 = Dense(128, activation='relu',kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2))
        self.f2 = Dense(64, activation='relu',kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2))
        self.d1 = Dropout(0.2)
        self.f3 = Dense(2, activation='linear',kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
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
    def __init__(self, alpha=0.05, weight=[0.5, 1, 0.5]):
        self.soften_ = 160.
        self.alpha = alpha
        self.lamda = 1e-2
        self.model = None
        self.weight = weight

    def loss_func(self, y_true, y_pred):

        pi_upper = y_pred[:, 0]
        pi_lower = y_pred[:, 1]

        y_T = tf.cast(y_true, dtype=tf.float32)

        N_ = tf.cast(tf.size(y_T), tf.float32)  # sample size
        K_HU = tf.maximum(0., tf.sign(pi_upper - y_T))
        K_HL = tf.maximum(0., tf.sign(y_T - pi_lower))
        K_H = tf.multiply(K_HU, K_HL)
        K_SU = tf.sigmoid(self.soften_ * (pi_upper - y_T))
        K_SL = tf.sigmoid(self.soften_ * (y_T - pi_lower))
        K_S = tf.multiply(K_SU, K_SL)

        pimw = tf.reduce_mean(tf.abs((pi_upper-pi_lower)/y_T))
        piad = tf.reduce_mean(tf.abs(y_T - (pi_upper+pi_lower)/2))
        picp_h = tf.reduce_mean(K_H)
        picp_s = tf.reduce_mean(K_S)
        loss = self.weight[1]*pimw + self.weight[0]*self.lamda * N_ / \
            (self.alpha*(1-self.alpha)) * \
            tf.maximum(0., (1-self.alpha) - picp_s) + self.weight[2]*piad
        return loss

    def fit(self, train_x, train_y, val_data=None, batch_size=256, epochs=1000):
        if val_data is not None:
            val_x, val_y = val_data
            val_db = tf.data.Dataset.from_tensor_slices((val_x, val_y))
            val_db = val_db.batch(batch_size)

        ini_bias_up = np.percentile(train_y, 95)*1.1
        ini_bias_down = np.percentile(train_y, 5)/1.1
        self.model = PI_model(ini_bias_up, ini_bias_down)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-2, decay = 1e-2)

        train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_db = train_db.shuffle(batch_size*5).batch(batch_size)

        picp = []
        pimw = []
        piad = []
        loss = []
        val_loss = []
        for epoch in tqdm(range(epochs)):
            for step, (x, y) in enumerate(train_db):
                with tf.GradientTape() as tape:  # 梯度带中的变量为trainable_variables，可自动进行求导
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
                    # y = tf.cast(y, dtype=tf.float32)
                    # model_piei = Piei(val_out, y, (1-self.alpha),
                    #                     range(len(y)), range(len(y)))
                    # picp_ = model_piei.picp
                    # pimw_ = model_piei.pimw
                    # piad_ = model_piei.piad
                    # picp.append(picp_)
                    # pimw.append(pimw_)
                    # piad.append(piad_)
                val_loss.append(vali_loss)

                    # print(r"epoch={},loss={},val_loss={}".format(
                    #     epoch, train_loss, vali_loss))
                    # print(model_piei.print_cpmwad())
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

        plt.subplot(121)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')

        plt.subplot(122)
        plt.plot(picp, label='Validation picp')
        plt.plot(pimw, label='Validation pimw')
        plt.plot(piad, label='Validation piad')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    def predict(self, test_x):
        result = self.model.predict(test_x)
        return result




if __name__ == '__main__':
    import time
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    train_data, target = load_boston(return_X_y=True)
    min_max_scaler = MinMaxScaler()
    train_data = min_max_scaler.fit_transform(train_data)
    train_X, test_X, train_y, test_y = train_test_split(train_data, target, test_size=0.2,
                                                        random_state=int(time.time()))
    train_XX, val_x, tar, val_y = train_test_split(train_X, train_y, test_size=0.2,
                                                   random_state=int(time.time()))
    alpha = 0.05

    tf_pi = TF_PI(alpha, weight=[1, 2, 1])
    history = tf_pi.fit(train_XX, tar, (val_x, val_y))
    result = tf_pi.predict(test_X)
    tf_pi.show_fit_process(history)
    model_pieitfpi = Piei(result, test_y, (1-alpha),
                          list(range(test_X.shape[0])), list(range(test_X.shape[0])))

    print("PICP = {}, PIMW = {}, PIAD = {}, PIEI = {}".format(model_pieitfpi.picp, model_pieitfpi.pimw,
                                                              model_pieitfpi.piad, model_pieitfpi.result_piei))
    from utils.plot_func import PlotErrorBar

    PlotErrorBar(result, list(
        range(test_X.shape[0])), test_y, model_pieitfpi.outlier)
