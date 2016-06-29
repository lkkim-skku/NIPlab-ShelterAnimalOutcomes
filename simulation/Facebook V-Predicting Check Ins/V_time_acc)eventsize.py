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
    for row in queryresult:
        acc, t = row[0], row[1]
        plt.plot(t, acc, 'b+')
        i += 1
        if i % 1000000 == 0:
            fig.savefig('acc_time_{}.png'.format(repr(i)), dpi=100)
            plt.clf()
            fig = plt.gcf()
            print(i, '번째 row 처리')
    fig.savefig('accuracy_time_{}.png'.format(repr(i)), dpi=100)
    print(i, '번째 row 처리')


if __name__ == '__main__':
    projname = os.path.split(sys.path[0])[-1]
    conn = sqlite3.connect(os.path.join(kaggle.path_project(projname), 'train.db'))
    accuracy_time(conn.cursor())

    conn.close()
