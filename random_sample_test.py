import numpy as np
from models.tensorflow_pi import TF_PI
from utils.tools import Piei
from utils.plot_func import plot_simple_boundary

# create some data
def gen_data(n=50, bound=1, deg=3, beta=1, noise=0.9, intcpt=-1):
    x = np.linspace(-bound, bound, n)[:, np.newaxis]
    h = np.linspace(-bound, bound, n)[:, np.newaxis]
    e = np.random.randn(*x.shape) * (0.1 + 10 * np.abs(x))
    y = 50 * (x ** deg) + h * beta + noise * e + intcpt
    return x, y.squeeze()

n_samples = 300
x, y = gen_data(n_samples, noise=1.0)

alpha = 0.1
tf_pi = TF_PI(alpha, weight=[3, 1, 1])
history = tf_pi.fit(x, y, epochs=1000)
result = tf_pi.predict(x)
model_pieitfpi = Piei(result, y, (1-alpha),
                      range(len(x)), range(len(x)))

print("PICP = {}, PIMW = {}, PIAD = {}, PIEI = {}".format(model_pieitfpi.picp, model_pieitfpi.pimw,
                                                          model_pieitfpi.piad, model_pieitfpi.result_piei))
plot_simple_boundary(result, x.squeeze(), y)