#  TOC_data well_3
import numpy as np

from models.ensemble_pi import Bootstrap_PI
from utils.data_preprocess import pre_process
from utils.plot_func import (plot_multi_boundary, plot_pi_toc,
                             plot_simple_boundary)
from utils.tools import Piei

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


alpha = 0.1
## model = Bootstrap_PI(10, alpha, bootstrap_method='replace_resample')
model = Bootstrap_PI(5, alpha, [1, 0.5, 0.5], bootstrap_method='prop_of_data')
result_all, hist, y_pred_gauss_mid, y_pred_gauss_dev, up_low = model.fit_predict(
    X, y, test_data)

model_pieitoc = Piei(up_low, y, (1-alpha),
                     toc_data['DEPT'].to_numpy(), logging_data['DEPT'].to_numpy())
print("PICP = {}, PIMW = {}, PIAD = {}, PIEI = {}".format(model_pieitoc.picp, model_pieitoc.pimw,
                                                          model_pieitoc.piad, model_pieitoc.result_piei))


plot_pi_toc(up_low, merge_toc, logging_data, stratum_depth,
            stratum_name, model_pieitoc.outlier, y_pred_gauss_dev)
