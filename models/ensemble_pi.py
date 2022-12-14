import copy
import time

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from models.tensorflow_pi import TF_PI
from utils.tools import Loss_func, all_to_pi, cal_pi_index


class Bootstrap_PI():
    def __init__(self, n_ensemble, alphas, weight=[1, 0.5, 0.5], epochs = 1000, bootstrap_method='prop_of_data', prop_select=0.80):
        self.n_ensemble = n_ensemble
        self.bootstrap_method = bootstrap_method
        self.prop_select = prop_select
        if n_ensemble > 1 and isinstance(alphas,list) != 1:
            self.alphas = [alphas]*self.n_ensemble
        elif n_ensemble == 1 and len([alphas]) == 1:
            self.alphas = [alphas]
        else:
            self.alphas = alphas
        self.weight = np.array(weight)
        self.epochs = epochs

    def fit_predict(self, train_x, train_y, test_X, val_data=None):
        self.train_x, self.train_y = copy.deepcopy(
            train_x), copy.deepcopy(train_y)
        hist = []
        result_all = []
        for i in tqdm(range(self.n_ensemble)):
            np.random.seed(int(time.time()))
            if self.bootstrap_method == 'replace_resample':
                # resample w replacement method
                id = np.random.choice(
                    train_x.shape[0], train_x.shape[0], replace=True)
                fit_x = self.train_x[id]
                fit_y = self.train_y[id]

            elif self.bootstrap_method == 'prop_of_data':
                # select x% of data each time NO resampling
                perm = np.random.permutation(train_x.shape[0])
                fit_x = self.train_x[perm[:int(
                    perm.shape[0] * self.prop_select)]]
                fit_y = self.train_y[perm[:int(
                    perm.shape[0] * self.prop_select)]]
                val_x = self.train_x[perm[int(
                    perm.shape[0] * self.prop_select):]]
                val_y = self.train_y[perm[int(
                    perm.shape[0] * self.prop_select):]]
                val_data = (val_x,val_y)
            elif self.bootstrap_method == 'ori_data':            
                fit_x = self.train_x
                fit_y = self.train_y
            tf_pi = TF_PI(self.alphas[i], weight=self.weight)
            if val_data:
                pass
            else:
                perm = np.random.permutation(train_x.shape[0])
                val_x = self.train_x[perm[:int(
                    perm.shape[0] * 0.2)]]
                val_y = self.train_y[perm[:int(
                    perm.shape[0] * 0.2)]]           
            history = tf_pi.fit(fit_x, fit_y, (val_x, val_y), epochs = self.epochs)
            result = tf_pi.predict(test_X)
            hist.append(history)
            result_all.append(result)
        result_all = np.array(result_all)
        y_pred_gauss_mid, y_pred_gauss_dev, up_low = all_to_pi(
            result_all, style='average')
        return result_all, hist, y_pred_gauss_mid, y_pred_gauss_dev, up_low


def diff_alphas_PI(train_x, train_y, test_x, sample_position, pre_position, alphas, weight=[1, 0.5, 0.5], bootstrap_method='ori_data'):
    
    model = Bootstrap_PI(len(alphas), alphas,  weight=weight, epochs = 1000,
                         bootstrap_method=bootstrap_method)
    result_all, hist, _, _, _ = model.fit_predict(train_x, train_y, test_x)
    index_all, outlier_list = cal_pi_index(result_all, train_y, alphas,
                             sample_position, pre_position, weight=weight)

    return result_all, index_all, outlier_list, hist


if __name__ == '__main__':

    import numpy as np

    from models.ensemble_pi import Bootstrap_PI
    from utils.data_preprocess import pre_process
    from utils.plot_func import (plot_multi_boundary, plot_pi_toc,
                                 plot_simple_boundary)
    from utils.tools import Loss_func

    TOC_file = './data/well_3/TOC_data_liushagang_2.csv'
    # TOC_file = './data/well_3/TOC_data.csv'
    welllog_file = './data/well_3/welllog_data.csv'

    stratum_depth = [2402.4, 2543.3, 2790.3, 2995]
    stratum_name = ['Liushagang_1', 'Liushagang_2', 'Liushagang_3']
    logging_data, toc_data, unit, merge_toc, _ = pre_process(
        TOC_file, welllog_file, stratum_depth)
    X = merge_toc[merge_toc.columns.difference(['DEPT', 'TOC'])].to_numpy()
    y = merge_toc['TOC'].to_numpy()
    test_data = logging_data[logging_data.columns.difference(['DEPT'])]

    alpha = 0.05
    ## model = Bootstrap_PI(10, alpha, bootstrap_method='replace_resample')
    model = Bootstrap_PI(1, alpha, weight=[0.5, 1, 0.5],
                         bootstrap_method='prop_of_data')
    result_all, hist, y_pred_gauss_mid, y_pred_gauss_dev, up_low = model.fit_predict(
        X, y, test_data)

    model_loss = Loss_func(up_low, y, (1-alpha),
                         toc_data['DEPT'].to_numpy(), logging_data['DEPT'].to_numpy())
    print("PICP = {}, PIMW = {}, PIAD = {}, Loss = {}".format(model_loss.picp, model_loss.pimw,
                                                              model_loss.piad, model_loss.loss))

    plot_pi_toc(up_low, merge_toc, logging_data, stratum_depth,
                stratum_name, model_loss.outlier, y_pred_gauss_dev)
