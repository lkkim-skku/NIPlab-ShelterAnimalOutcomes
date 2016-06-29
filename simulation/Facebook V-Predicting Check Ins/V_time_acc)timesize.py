import os
import sys
import sqlite3
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../'))
import kaggle


def accuracy_time(cursor: sqlite3.Cursor):
    queryresult = cursor.execute('''SELECT DISTINCT accuracy, time from train ORDER BY time''')
    i = 0
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    last_t = 0
    for row in queryresult:
        acc, t = row[0], row[1]
        plt.plot(t, acc, 'b+')
        last_t = t
        if t > 0 and t % 500 == 0:
            plt.title('time')
            fig.savefig('time_accuracy_{}.png'.format('%10d' % (t - 1)), dpi=100)
            plt.clf()
            fig = plt.gcf()
            print(t, '까지의 image 처리')
    fig.savefig('time_accuracy_{}.png'.format('%10d' % last_t), dpi=100)
    print(last_t, '까지의 image 처리')


if __name__ == '__main__':
    projname = os.path.split(sys.path[0])[-1]
    conn = sqlite3.connect(os.path.join(kaggle.path_project(projname), 'train.db'))
    accuracy_time(conn.cursor())

    conn.close()
