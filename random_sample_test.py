import numpy as np
from models.tensorflow_pi import TF_PI
from utils.tools import Piei
from utils.plot_func import plot_simple_boundary

# create some data
n_samples = 100
X = np.random.uniform(low=-2.,high=2.,size=(n_samples,1))
y = 1.5*np.sin(np.pi*X[:,0]) + np.random.normal(loc=0.,scale=1.*np.power(X[:,0],2))
y = y.reshape([-1,1])
X_train = X.reshape(-1)
y_train = y.reshape(-1)
y_train = np.stack((y_train,y_train),axis=1) # make this 2d so will be accepted
x_grid = np.linspace(-2,2,100) # for evaluation plots


alpha = 0.1
tf_pi = TF_PI(alpha, weight=[1, 1, 1])
history = tf_pi.fit(X_train, y_train, epochs=1000)
result = tf_pi.predict(X_train)
model_pieitfpi = Piei(result, y, (1-alpha),
                      range(len(X)), range(len(X)))

print("PICP = {}, PIMW = {}, PIAD = {}, PIEI = {}".format(model_pieitfpi.picp, model_pieitfpi.pimw,
                                                          model_pieitfpi.piad, model_pieitfpi.result_piei))
plot_simple_boundary(result, X.squeeze(), y)