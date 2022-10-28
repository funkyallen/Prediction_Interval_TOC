from models.ensemble_pi import multi_alphas_PI
from utils.plot_func import subplot_multi_boundary,subplot_fit_process
from utils.data_preprocess import pre_process

TOC_file = './data/well_3/TOC_data_liushagang_2.csv'
# TOC_file = './data/well_3/TOC_data_liushagang.csv'
# TOC_file = './data/well_3/TOC_data.csv'
welllog_file = './data/well_3/welllog_data.csv'

stratum_depth = [2402.4, 2543.3, 2790.3, 2995]
stratum_name = ['Liushagang_1', 'Liushagang_2', 'Liushagang_3']
logging_data, toc_data, unit, merge_toc, _ = pre_process(
    TOC_file, welllog_file, stratum_depth)
X = merge_toc[merge_toc.columns.difference(['DEPT', 'TOC'])].to_numpy()
y = merge_toc['TOC'].to_numpy()
test_data = logging_data[logging_data.columns.difference(['DEPT'])]
alphas = [0.2, 0.15, 0.1, 0.05]
result_all, index_all, outlier_list, hist = multi_alphas_PI(
    X, y, test_data, merge_toc['DEPT'].to_numpy(), logging_data['DEPT'].to_numpy(), alphas, weight=[1, 1, 1], bootstrap_method='prop_of_data')


# plot_multi_boundary(result_all, alphas, toc_data['DEPT'].to_numpy(
# ), logging_data['DEPT'].to_numpy(), y)
print(index_all)
subplot_multi_boundary(result_all, alphas, outlier_list,
                       merge_toc['DEPT'].to_numpy(), logging_data['DEPT'].to_numpy(), y)

subplot_fit_process(hist, alphas)