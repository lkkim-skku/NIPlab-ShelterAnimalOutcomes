import os
import sys
import sqlite3
import csv
sys.path.append(os.path.abspath('../'))
import kaggle


def createtable(conn):
    csrr = conn.cursor()
    csrr.execute('''DROP TABLE IF EXISTS train''')
    csrr.execute('''CREATE TABLE IF NOT EXISTS train (x REAL, y REAL, accuracy INTEGER, time INTEGER, place_id VARCHAR(10))''')

    print('<train>')
    with open(os.path.join(kaggle.path_project(projname), 'train.csv'), 'r') as file:
        csver = csv.reader(file, delimiter=',', quotechar='|')
        header = next(csver)
        i = 0
        for row in csver:
            # print(row)
            csrr.execute('''INSERT INTO train VALUES ({}, {}, {}, {}, {})'''.format(float(row[1]), float(row[2]), int(row[3]), int(row[4]), row[5]))
            i += 1
            if i % 10000 == 0:
                print(i, "번째 줄 commit")
                conn.commit()
        conn.commit()
        print(i, '개 row commit 완료')

    csrr.execute('''CREATE INDEX class ON train (place_id)''')
    csrr.execute('''CREATE INDEX coord ON train (x, y)''') 


if __name__ == '__main__':
    projname = os.path.split(sys.path[0])[-1]
    conn = sqlite3.connect(os.path.join(kaggle.path_project(projname), 'train.db'))
    csrr = conn.cursor()

    createtable(conn)

    conn.close()
