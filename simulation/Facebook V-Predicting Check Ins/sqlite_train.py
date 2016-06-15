import os
import sys
import sqlite3
import csv
sys.path.append(os.path.abspath('../'))
import kaggle


def createtable(conn):
    csrr = conn.cursor()
    csrr.execute('''DROP TABLE IF EXISTS train''')
    csrr.execute('''CREATE TABLE IF NOT EXISTS train (row_id integer PRIMARY KEY, x integer, y integer, accuracy integer, time integer, place_id integer)''')

    print('<train>')
    with open(os.path.join(kaggle.path_project(projname), 'train.csv'), 'r') as file:
        csver = csv.reader(file, delimiter=',', quotechar='|')
        header = next(csver)
        i = 0
        for row in csver:
            csrr.execute('''INSERT INTO train VALUES ({}, {}, {}, {}, {}, {})'''.format(row[0], row[1], row[2], row[3], row[4], row[5]))
            i += 1
            if i % 10000 == 0:
                print(i, "번째 줄 commit")
                conn.commit()
        conn.commit()
        print(i, '개 row commit 완료')


if __name__ == '__main__':
    projname = os.path.split(sys.path[0])[-1]
    conn = sqlite3.connect(os.path.join(kaggle.path_project(projname), 'train.db'))
    csrr = conn.cursor()

    createtable(conn)

    conn.close()
