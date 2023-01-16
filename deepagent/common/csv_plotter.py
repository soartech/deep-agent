from past.utils import old_div
import csv

from matplotlib import pyplot as pl


def plot_it(csv_file):
    x = []
    y = []
    y_low = []
    y_high = []
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if first:
                first = False
                continue
            x.append(float(row[0]))
            y.append(old_div(float(row[1]), 15.0))
            y_low.append(old_div(float(row[2]), 15.0))
            y_high.append(old_div(float(row[3]), 15.0))
    win = [.84 for _ in range(len(x))]
    pl.title('Image State, Leaky Activation')
    pl.xlabel('Game Number')
    pl.ylabel('Discounted Reward')
    pl.xlim((0, 35))
    pl.ylim((-0.5, 2.0))
    pl.plot(x, y, 'k-')
    pl.plot(x, win, 'k-')
    # pl.fill_between(x, y_low, y_high)
    pl.show()
    pl.savefig('test.png')


if __name__ == '__main__':
    plot_it(csv_file='15v16.csv')
