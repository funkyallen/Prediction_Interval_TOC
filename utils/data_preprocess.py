import copy

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import spatial
from sklearn import preprocessing

matplotlib.use('TkAgg')


def fuzzy_merge(logging_data, toc_data):
    """
    Use kdtree to merge toc_data and logging_data by depth.
    :param logging_data:
    :param toc_data:
    :return:
    """
    depth_log = logging_data['DEPT'].to_numpy()
    depth_toc = toc_data['DEPT'].to_numpy()
    depth_log = [[depth_log[i], depth_log[i]] for i in range(len(depth_log))]
    depth_toc = [[depth_toc[i], depth_toc[i]] for i in range(len(depth_toc))]
    tree = spatial.KDTree(depth_log)
    _, _seq = tree.query(depth_toc)
    merge_toc = logging_data.iloc[_seq, :]
    merge_toc = merge_toc.drop_duplicates(subset='DEPT')
    merge_toc = pd.merge(merge_toc, toc_data, how='inner', on='DEPT')
    return merge_toc


def depth_merge(logging_data, toc_data):
    depth_log = logging_data['DEPT'].to_numpy()
    depth_toc = toc_data['DEPT'].to_numpy()
    i = 0  # toc sequence
    j = 0  # log sequence
    _seq = []
    while True:
        if i == len(depth_toc):
            break
        if j == len(depth_log):
            break
        if abs(depth_log[j] - depth_toc[i]) < 0.08:
            _seq.append(j)
            i += 1
            j += 1
        else:
            j += 1
    merge_toc = logging_data.iloc[_seq, :]
    merge_toc.loc[:, 'TOC'] = toc_data['TOC'].values
    return merge_toc


def pre_process(TOC_file, welllog_file, stratum_depth):
    """
    :param TOC_file:
    :param welllog_file:
    :return: training_data and labels for machine learning
    """
    toc_data = pd.read_csv(TOC_file, skiprows=[0])
    if toc_data.shape[1] == 3:
        toc_data.columns = ['DEPT', 'TOC', 'STRATUM']
    else:
        toc_data.columns = ['DEPT', 'TOC']
    logging_data = pd.read_csv(welllog_file, skiprows=[1])
    # Get the unit of each logging parameters
    unit = pd.read_csv(welllog_file, nrows=2, header=None)
    unit = unit.iloc[1, :-1]
    # Data clean
    columns_list = logging_data.columns.tolist()
    for column in columns_list:
        logging_data = logging_data[~logging_data[column].isin([-999.25])]

    # toc_data = toc_data[~toc_data['DEPT'] < logging_data.DEPT.values[0]]
    toc_data = toc_data.drop(
        toc_data[toc_data['DEPT'] < logging_data.DEPT.values[0]].index)
    toc_data = toc_data.drop(
        toc_data[toc_data['DEPT'] > logging_data.DEPT.values[-1]].index)
    logging_data = logging_data.drop(
        logging_data[logging_data['DEPT'] < 2000].index)
    logging_data['DEPT'] = logging_data['DEPT'].round(1)
    toc_data['DEPT'] = toc_data['DEPT'].round(1)

    def stratum(x):
        for i in range(len(stratum_depth)-1):
            if x >= stratum_depth[i] and x < stratum_depth[i+1]:
                return int(i)
    logging_data['STRATUM'] = logging_data['DEPT'].apply(stratum)
    logging_data = logging_data.drop(
        logging_data[logging_data['STRATUM'].isnull()].index)
    logging_data = pd.get_dummies(logging_data, columns=['STRATUM'])
    ori_data = logging_data.copy(deep=True)
    # TOC merge with logging data
    cols = logging_data.columns.difference(["DEPT"])
    min_max_scaler = preprocessing.RobustScaler()
    logging_data[cols] = min_max_scaler.fit_transform(logging_data[cols])
    merge_toc = fuzzy_merge(logging_data, toc_data)
    return logging_data, toc_data, unit, merge_toc, ori_data


def plot(merge_toc):
    sns.set_theme(style="ticks")
    # sns.pairplot(merge_toc, hue='TOC')
    sns.heatmap(merge_toc.corr(), annot=True, cmap="YlGnBu")
    plt.show()


if __name__ == '__main__':
    # TOC_file = './data/well_1/TOC_data.csv'
    # welllog_file = './data/well_1/welllog_data.csv'
    TOC_file = './data/well_3/TOC_data.csv'
    # welllog_file = './data/well_3/welllog_data.csv'
    # TOC_file = './data/well_3/TOC_data_liushagang_2.csv'
    welllog_file = './data/well_3/welllog_data.csv'
    stratum_depth = [2402.4, 2543.3, 2790.3, 2995]
    logging_data, toc_data, unit, merge_toc, ori_data = pre_process(
        TOC_file, welllog_file, stratum_depth)

    # plot(merge_toc)
