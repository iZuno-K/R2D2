import csv
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np


def smooth_plot(x_s, y_s, interval):
    """smooth plot by averaging"""
    sta = 0
    x = []
    y = []
    for i in range(int(len(x_s) / interval)):
        x.append(np.mean(x_s[sta: sta + interval]))
        y.append(np.mean(y_s[sta: sta + interval]))
        sta += interval

    return x, y

def simple_csv_plotter(log_file, save_dir):
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        # header = next(reader)
        header = next(reader)

        data = [a for a in reader]
        data = list(zip(*data))  # [[1., 'a', '1h'], [2., 'b', '2b']] -> [(1., 2.), ('a', 'b'), ('1h', '2h')]
        data_dict = {header[i]: list(data[i]) for i in range(len(header))}

    # x_label = 'episodes'
    # ylabel = "r"
    # x_data = np.arange(len(data_dict[ylabel]))
    # y_datas = [float(i) for i in data_dict[ylabel]]
    # x, y = smooth_plot(x_data, y_datas, interval=1)

    x_label = 'episodes'
    ylabel = "mean 100 episode reward"
    x_data = [int(i) for i in data_dict[x_label]]
    y_datas = [float(i) for i in data_dict[ylabel]]
    x, y = smooth_plot(x_data, y_datas, interval=10)

    plt.style.use('mystyle2')
    fig = plt.figure()

    plt.plot(x, y)

    plt.title("Return curve")
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(save_dir, 'return_curve.pdf'))
    # plt.show()

def csv_log_plotter(log_file, save_dir):
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        data = [a for a in reader]
        data = list(zip(*data))  # [[1., 'a', '1h'], [2., 'b', '2b']] -> [(1., 2.), ('a', 'b'), ('1h', '2h')]
        data_dict = {header[i]: list(data[i]) for i in range(len(header))}

    x_label = 'episodes'
    y_labels = ['mean 100 episode reward', 'mean 100 episode loss']
    x_data = [int(i) for i in data_dict[x_label]]
    y_datas = [[float(i) for i in data_dict[ylabel]] for ylabel in y_labels]

    plt.style.use('mystyle2')
    fig, axes = plt.subplots(2)

    for i, label in enumerate(y_labels):
        # axes[int(i/2), i % 2].set_title(label)
        axes[i].set_ylabel(label)
        axes[i].set_xlabel(x_label)

        x, y = smooth_plot(x_data, y_datas[i], interval=10)
        axes[i].plot(x, y)

    fig.suptitle('Return Curve')
    plt.savefig(os.path.join(save_dir, 'return_curve.pdf'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default=None)
    parser.add_argument('--root-dirs', type=str, default=None, help="data root directories name separated by a `^`")
    parser.add_argument('--labels', type=str, default=None, help="label names separated by a `^`")
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()
    file = os.path.join(args['root_dir'], 'progress.csv')
    simple_csv_plotter(file, args['root_dir'])
    # csv_log_plotter(file, args['root_dir'])
