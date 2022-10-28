# -*- coding:utf-8 _*-
"""
@Author:Jianwen Rui
@E-mail: first91@163.com
@FileName: plot_func.py
@SoftWare: PI_TOC
@DateTime: 2022/7/6 0006 15:10
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from collections import OrderedDict
import matplotlib.pyplot as plt


def PlotErrorBar(result, x, target, outer_list):
    point_result = [((result[i, 0] + result[i, 1]) / 2)
                    for i in range(result.shape[0])]
    error = [abs(result[i, 0] - result[i, 1]) /
             2 for i in range(result.shape[0])]
    plt.figure(int(time.time()))
    wid = len(target) / 100
    hei = np.percentile(result, 50) / 100
    from matplotlib import patches
    from collections import OrderedDict
    plt.errorbar(x, point_result, yerr=error, fmt='none',
                 elinewidth=1.5, ms=3, mfc='wheat', mec='salmon', capsize=3)
    plt.scatter(x, target, label='Testing dataset', color='darkorange', s=65)

    for i in range(len(outer_list)):
        if i == 0:
            plt.gca().add_patch(
                patches.Ellipse(xy=(x[outer_list[i]], target[outer_list[i]]), width=wid, height=hei, fill=False, linewidth=2,
                                color='m', label='Outer point'))
        else:
            plt.gca().add_patch(
                patches.Ellipse(xy=(x[outer_list[i]], target[outer_list[i]]), width=wid, height=hei, fill=False, linewidth=2,
                                color='m'))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.tick_params(labelsize=20)
    plt.xlabel('Sample', fontsize=23)
    plt.ylabel('Scaled sound pressure level', fontsize=23)
    plt.legend(by_label.values(), by_label.keys(), fontsize=21)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.show()


def plot_pi_toc(up_low, merge_toc, logging_data, stratum_depth, stratum_name, list_train, model_std=None):
    from matplotlib import patches
    from collections import OrderedDict
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.hlines(2, logging_data['DEPT'].iloc[0], logging_data['DEPT'].iloc[-1], linestyles='dashed', color='dimgray',
              label='TOC content=2')
    plot_1 = ax.fill_between(np.array(logging_data['DEPT'].values), up_low[:, 1], up_low[:, 0], color='powderblue',
                             label='Prediction interval')
    plot_2, = ax.plot(np.array(logging_data['DEPT'].values), (up_low[:, 0] + up_low[:, 1]) / 2, color='royalblue',
                      label='Interval middle line', linewidth=3)
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Depth (m)', fontsize=23)
    ax.set_ylabel('TOC content (wt.%)', fontsize=23)
    for i in range(1, len(stratum_depth)-1):
        ax.axvline(stratum_depth[i], ls='--')
    for i in range(len(stratum_name)):
        ax.text((stratum_depth[i]+stratum_depth[i+1]) /
                3, 4.5, stratum_name[i], fontsize=21)

    plot_3 = ax.scatter(merge_toc['DEPT'], merge_toc['TOC'],
                        label='Training point', color='darkorange', s=65)
    if list_train:
        for i in range(len(list_train)):
            if i == 0:
                plot_4 = plt.gca().add_patch(
                    patches.Ellipse(xy=(merge_toc.iloc[list_train[i], 0], merge_toc.iloc[list_train[i], -1]),
                                    width=logging_data.shape[0] / 400,
                                    height=(max(up_low[:, 0]) -
                                            min(up_low[:, 0])) / 5,
                                    fill=False, linewidth=2, color='m', label='Outer point'))
            else:
                plt.gca().add_patch(
                    patches.Ellipse(xy=(merge_toc.iloc[list_train[i], 0], merge_toc.iloc[list_train[i], -1]),
                                    width=logging_data.shape[0] / 400,
                                    height=(max(up_low[:, 0]) -
                                            min(up_low[:, 0])) / 5,
                                    fill=False, linewidth=2, color='m'))
    if model_std is not None:
        ax2 = ax.twinx()
        plot_5, = ax2.plot(np.array(
            logging_data['DEPT'].values), model_std, color='#C82423', ls='-.', label='Uncertainty')
        ax2.tick_params(labelsize=20)
        ax2.set_ylabel('Uncertainty', fontsize=23)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), fontsize=21)
    if model_std is not None:
        plt.legend([plot_1, plot_2, plot_3, plot_4, plot_5], ['Prediction interval',
                                                              'Interval middle line', 'Test data points', 'Outer point', 'Uncertainty'], fontsize=20)
    else:
        plt.legend([plot_1, plot_2, plot_3, plot_4], ['Prediction interval',
                                                      'Interval middle line', 'Test data points', 'Outer point'], fontsize=20)
    plt.savefig('./images/{}TOC_fill.png'.format(str(int(time.time()))[-5:]))
    plt.show()


def plot_simple_boundary(up_low, x, test_y):

    fig = plt.figure(figsize=(25, 16))
    ax = fig.add_subplot()

    plot_2 = ax.fill_between(
        x, up_low[:, 1], up_low[:, 0], color='powderblue',  label='Prediction interval')

    plot_4 = ax.scatter(x, test_y, color='darkorange',
                        label='Test data points')
    ax.set_xlabel('X', fontsize=23)
    ax.set_ylabel('Y', fontsize=23)
    ax.tick_params(labelsize=20)

    fig.tight_layout()
    # handles, labels = plt.gca().get_legend_handles_labels()  #get_legend_handles_labels
    # by_label = OrderedDict(zip(labels, handles)) # remove dupulicated labels
    # ax.legend(by_label.values(), by_label.keys(), fontsize=21)
    plt.legend([plot_2, plot_4], ['Prediction interval',
               'Test data points'], fontsize=20)
    plt.show()


def plot_boundary(up_low, y_pred_all, y_pred_gauss_mid, y_pred_gauss_dev, x, test_y):
    color = ['cyan', 'aqua', 'violet', 'blue']
    fig = plt.figure(figsize=(25, 16))
    ax = fig.add_subplot()
    for i in range(y_pred_all.shape[0]):
        plot_1, = ax.plot(x, y_pred_all[i, :, 0], c='g', alpha=0.5,
                          linestyle='--', linewidth=0.5, label='Indiv. boundary')
        ax.plot(x, y_pred_all[i, :, 1], c='g',
                alpha=0.5, linestyle='--', linewidth=0.5)
    plot_2 = ax.fill_between(
        x, up_low[:, 1], up_low[:, 0], color='powderblue',  label='Prediction interval')
    plot_3, = ax.plot(x, y_pred_gauss_mid, color='royalblue',
                      label='Interval middle line', linewidth=3)
    plot_4 = ax.scatter(x, test_y, color='darkorange',
                        label='Test data points')
    ax.set_xlabel('X', fontsize=23)
    ax.set_ylabel('Y', fontsize=23)
    ax.tick_params(labelsize=20)
    ax2 = ax.twinx()
    plot_5, = ax2.plot(x, y_pred_gauss_dev, alpha=0.6, linestyle='-.',
                       color='#C82423', linewidth=1, label='Uncertainty')
    ax2.set_ylabel('Uncertainty', fontsize=23)
    ax2.tick_params(labelsize=20)
    ax2.legend(fontsize=21)
    fig.tight_layout()
    # handles, labels = plt.gca().get_legend_handles_labels()  #get_legend_handles_labels
    # by_label = OrderedDict(zip(labels, handles)) # remove dupulicated labels
    # ax.legend(by_label.values(), by_label.keys(), fontsize=21)
    plt.legend([plot_1, plot_2, plot_3, plot_4, plot_5], ['Indiv. boundary', 'Prediction interval',
               'Interval middle line', 'Test data points', 'Uncertainty'], fontsize=20)
    plt.savefig('images/ENSEM_TOC_{}.png'.format(str(int(time.time()))[6:]))
    plt.show()


def plot_multi_boundary(y_pred_all, alphas, sample_position, pre_position, test_y):
    color = ['cyan', 'aqua', 'violet', 'blue']
    fig = plt.figure(figsize=(25, 16))
    ax = fig.add_subplot()
    for i in range(y_pred_all.shape[0]):
        plot_1, = ax.fill_between(pre_position, y_pred_all[i, :, 0], y_pred_all[i, :, 1],
                                  color=color[i], label=r'$\alpha$={}'.format(alphas[i]), alpha=0.1*i+0.6)
    ax.scatter(sample_position, test_y, color='darkorange',
               label='Test data points')
    ax.set_xlabel('X', fontsize=23)
    ax.set_ylabel('Y', fontsize=23)
    ax.tick_params(labelsize=20)
    fig.tight_layout()
    # handles, labels = plt.gca().get_legend_handles_labels()  #get_legend_handles_labels
    # by_label = OrderedDict(zip(labels, handles)) # remove dupulicated labels
    # ax.legend(by_label.values(), by_label.keys(), fontsize=21)
    plt.legend(fontsize=20)
    # plt.savefig('images/ENSEM_TOC_{}.png'.format(str(int(time.time()))[6:]))
    plt.show()


def subplot_multi_boundary(y_pred_all, alphas, outlier_list, sample_position, pre_position, test_y):
    from matplotlib import patches
    fig = plt.figure(figsize=(16, 9))
    for i in range(y_pred_all.shape[0]):
        plt.subplot(2, 2, i+1)
        plt.hlines(2, pre_position[0], pre_position[-1],
                   linestyles='dashed', color='dimgray')
        for j in range(len(outlier_list[i])):
            plot_4 = plt.gca().add_patch(
                patches.Ellipse(xy=(sample_position[outlier_list[i][j]], test_y[outlier_list[i][j]]),
                                width=pre_position.shape[0] / 400,
                                height=(max(y_pred_all[i, :, 0]) -
                                        min(y_pred_all[i, :, 1])) / 5,
                                fill=False, linewidth=2, color='m'))
        midline = np.mean(y_pred_all[i, :], axis=1)
        plot_1 = plt.fill_between(pre_position, y_pred_all[i, :, 0], y_pred_all[i, :, 1],
                                  color='powderblue', label=r'$\alpha$={}'.format(alphas[i]))
        plot_2, = plt.plot(pre_position, midline, color='royalblue',
                           label='Interval midline')
        plot_3 = plt.scatter(sample_position, test_y, color='darkorange',
                             label='Test data points')
        if i >= 2:
            plt.xlabel('Depth(m)', fontsize=16)
        if i % 2 == 0:
            plt.ylabel('TOC content (wt.%)', fontsize=16)
        plt.ylim(0, 6)
        plt.tick_params(labelsize=15)
        fig.tight_layout()
        plt.title(r'$ \alpha=$''%.2f' % alphas[i], fontsize=19)
    # handles, labels = plt.gca().get_legend_handles_labels()  #get_legend_handles_labels
    # by_label = OrderedDict(zip(labels, handles)) # remove dupulicated labels
    # ax.legend(by_label.values(), by_label.keys(), fontsize=21)

    plt.legend([plot_1, plot_2, plot_3, plot_4], ['Prediction interval',
                                                  'Interval middle line', 'Test data points', 'Outer point'],
               ncol=4, bbox_to_anchor=(0.05, -0.2), loc=10, fontsize=16)
    # plt.savefig('images/ENSEM_TOC_{}.png'.format(str(int(time.time()))[6:]))
    plt.show()

def subplot_fit_process(hist, alphas):
    from matplotlib import patches
    fig = plt.figure(figsize=(16, 9))
    for i in range(len(alphas)):
        plt.subplot(2, 2, i+1)
        loss = hist[i].history['loss']
        val_loss = hist[i].history['val_loss']
        plt.plot(range(len(loss)),loss, label='Training Loss')
        if val_loss:
            plt.plot(range(len(val_loss)),val_loss, label='Validation Loss')
        plt.title(r'$\alpha$={}'.format(alphas[i]), fontsize=23)
        plt.tick_params(labelsize=17)
        if i % 2 == 0:
            plt.ylabel('Loss',fontsize=21)
        if i >= 2:
            plt.xlabel('Epochs',fontsize=21)
            
    plt.legend(ncol=2, bbox_to_anchor=(-0.1, -0.2), loc=10, fontsize=19)
    plt.show()    

def cross_plot(merge_toc, stratum_depth):

    def stratum(x):
        for i in range(len(stratum_depth)-1):
            if x >= stratum_depth[i] and x < stratum_depth[i+1]:
                return int(i)

    data = merge_toc.drop('DEPT', axis=1).values
    relation = []
    color = ['red', 'green', 'blue', 'orange']
    member = ['member 1', 'member 2', 'member 3', 'member 4']
    stratum_id = [stratum(merge_toc.iloc[j, 0]) for j in range(data.shape[0])]
    unit = ['US/FT', 'gAPI', 'OHMM', 'OHMM', 'mV', 'G/C3']
    for i in range(data.shape[1]-1):
        relation.append(np.corrcoef(data[:, i], data[:, -1])[0, 1].round(2))
        plt.subplot(2, 3, i+1)
        model = np.poly1d(np.polyfit(data[:, i], data[:, -1], deg=1).round(2))
        model_para = np.polyfit(data[:, i], data[:, -1], deg=1).round(2)
        plt.plot(np.linspace(min(data[:, i]), max(data[:, i]), 50), model(
            np.linspace(min(data[:, i]), max(data[:, i]), 50)), color='k', linewidth=1.5)
        plt.text(min(data[:, i])+(max(data[:, i])-min(data[:, i]))
                 * 0.05, 3.5, "y = {}x{}".format(model_para[0], model_para[1]), fontsize=18)
        plt.text(min(data[:, i])+(max(data[:, i])-min(data[:, i]))
                 * 0.05, 3.2, r'$R=$''%.2f' % relation[i], fontsize=18)
        for j in range(data.shape[0]):
            plt.scatter(data[j, i], data[j, -1],
                        color=color[stratum_id[j]], label=member[stratum_id[j]], alpha=0.8, s=50)
        plt.xlabel('{}({})'.format(list(merge_toc.columns.values)
                   [i+1], unit[i]), fontsize=21)
        plt.title("({}) {}—TOC".format(
            chr(97+i), list(merge_toc.columns.values)[i+1]), fontsize=23, fontstyle='italic')
        plt.ylabel('TOC (wt.%)', fontsize=21)
        plt.tick_params(labelsize=18)
        plt.ylim(0, 4)
        if i == 4:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), ncol=3,
                       bbox_to_anchor=(1.4, -0.15), fontsize=16)

        # if i == 4:
        #     plt.legend(ncol=3, bbox_to_anchor=(1.2, -0.15), fontsize=16)
    plt.show()


if __name__ == '__main__':
    from utils.data_preprocess import pre_process

    TOC_file = './data/well_3/TOC_data_liushagang_2.csv'
    # TOC_file = './data/well_3/TOC_data.csv'
    welllog_file = './data/well_3/welllog_data.csv'
    stratum_depth = [2402.4, 2543.3, 2790.3, 2995]
    logging_data, toc_data, unit, merge_toc, ori_data = pre_process(
        TOC_file, welllog_file, stratum_depth)
    drop_list = ['M2R1', 'M2R6', 'M2R9', 'M2RX',
                 'STRATUM_0.0', 'STRATUM_1.0', 'STRATUM_2.0']
    ori_logging = ori_data.drop(drop_list, axis=1)
    ori_merge = ori_logging.merge(toc_data, how='inner', on='DEPT')
    cross_plot(ori_merge, stratum_depth)
