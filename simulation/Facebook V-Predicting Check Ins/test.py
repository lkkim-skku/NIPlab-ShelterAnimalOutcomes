import os
import sys
import sqlite3
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../'))
import kaggle


def accuracy_time(cursor: sqlite3.Cursor):
    queryresult = cursor.execute('''SELECT accuracy, time from train''')
    plt.clf()
    for row in queryresult:
        acc, t = row[0], row[1]
        plt.plot(t, acc, 'b+')

    plt.show()
    pass


if __name__ == '__main__':
    projname = os.path.split(sys.path[0])[-1]
    conn = sqlite3.connect(os.path.join(kaggle.path_project(projname), 'train.db'))
    accuracy_time(conn.cursor())

    conn.close()
