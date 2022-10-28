import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def all_to_pi(y_pred_all, style='average'):
    in_ddof = 1 if y_pred_all.shape[0] > 1 else 0
    if style == 'conservative':
        y_pred_U = np.max(y_pred_all[:, :, 0], axis=0)
        y_pred_L = np.min(y_pred_all[:, :, 1], axis=0)

    elif style == 'average':
        y_pred_U = np.mean(y_pred_all[:, :, 0], axis=0)
        y_pred_L = np.mean(y_pred_all[:, :, 1], axis=0)

    y_pred_U_mean = np.mean(y_pred_all[:, :, 0], axis=0)
    y_pred_gauss_dev = np.std(y_pred_all[:, :, 0], axis=0)
    # y_pred_gauss_dev = np.sqrt(
    #     np.sum((y_pred_all[:, :, 0] - y_pred_U_mean)**2, axis=0)/(len(y_pred_all[:, :, 0])-1))
    y_pred_gauss_mid = np.mean((y_pred_U, y_pred_L), axis=0)
    # y_pred_gauss_dev = np.sqrt(
    #     (y_pred_U - y_pred_gauss_mid)**2 / (len(y_pred_all[:, :, 0])-1))

    up_low = np.vstack((y_pred_U, y_pred_L)).T

    return y_pred_gauss_mid, y_pred_gauss_dev, up_low


class Piei:
    def __init__(self, up_low, target, miu, sample_position, pre_position, weight=[1, 1, 1]):
        self.pi = up_low
        self.target = target
        self.miu = miu
        self.lamda = 1
        self.batch_size = 128 if len(
            sample_position) > 128 else len(sample_position)
        self.picp = None
        self.pimw = None
        self.piad = None
        self.weight = weight
        self.sample_position = sample_position
        self.pre_position = pre_position
        self.position = self.merge_position()
        self.outlier, self.picp, self.pimw, self.piad = self.cal_cp_mw_ad()
        self.result_piei = self.cal_piei()

    def merge_position(self):
        """
        Find the position of the given value. For example, core sample depth in all logging depth.
        """
        from scipy import spatial
        sample_position = [[self.sample_position[i], self.sample_position[i]]
                           for i in range(len(self.sample_position))]
        pre_position = [[self.pre_position[i], self.pre_position[i]]
                        for i in range(len(self.pre_position))]
        tree = spatial.KDTree(pre_position)
        _, seq = tree.query(sample_position)
        return seq.astype('int')

    def cal_cp_mw_ad(self):
        '''
        Calculate the coverage probability , mean width  and deviation of PI.
        '''
        list = []
        piad = 0
        pimw = 0
        for i in range(len(self.target)):

            pimw += abs((self.pi[self.position[i], 0] -
                        self.pi[self.position[i], 1]) / self.target[i])
            piad += abs(self.target[i] - (self.pi[self.position[i],
                        0] + self.pi[self.position[i], 1]) / 2)
            if (self.target[i] - self.pi[self.position[i], 0]) * (self.target[i] - self.pi[self.position[i], 1]) > 0:
                # if self.target[i] > self.pi[self.position[i], 0] or self.target[i] < self.pi[self.position[i], 1]:
                list.append(i)
        picp = round(1 - len(list) / len(self.target), 2)
        return list, picp, np.round(pimw / len(self.target), 2), np.round(piad / len(self.target), 2)

    def print_outlier(self):
        '''
        Print outlier information : target value, PI value.
        '''
        for i in range(len(self.outlier)):
            print("target={},PI={}".format(
                self.target[self.outlier[i]], self.pi[self.outlier[i], :]))

    def print_cpmwad(self):
        print('PICP={},PIMW={},PIAD={}'.format(
            self.picp, self.pimw, self.piad))

    def cal_piei(self):
        piei = self.weight[2]*self.piad + self.weight[1]*self.pimw + self.weight[0]*self.lamda*self.batch_size / \
            (self.miu*(1-self.miu))*(self.miu - self.picp)**2
        return np.round(piei, 2)


def cal_pi_index(result_all, target, alphas, sample_position, pre_position, weight=[1, 1, 1]):
    index_all = []
    outlier_list = []
    for i in range(result_all.shape[0]):
        model_pieitfpi = Piei(result_all[i, :], target, (1-alphas[i]),
                              sample_position, pre_position)
        index_all.append(np.array([model_pieitfpi.picp, model_pieitfpi.pimw,
                                  model_pieitfpi.piad, model_pieitfpi.result_piei]))
        outlier_list.append(model_pieitfpi.outlier)
    return index_all, outlier_list
