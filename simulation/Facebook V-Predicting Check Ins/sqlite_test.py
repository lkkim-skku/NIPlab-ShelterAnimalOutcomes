import os
import sys
import sqlite3
import csv
sys.path.append(os.path.abspath('../'))
import kaggle


def createtable(conn):
    crsr = conn.cursor()
    crsr.execute('''DROP TABLE IF EXISTS test''')
    crsr.execute('''CREATE TABLE IF NOT EXISTS test (row_id INTEGER PRIMARY KEY, x REAL, y REAL, accuracy INTEGER , time INTEGER)''')
    print('<test>')
    with open(os.path.join(kaggle.path_project(projname), 'test.csv'), 'r') as file:
        csver = csv.reader(file, delimiter=',', quotechar='|')
        header = next(csver)
        i = 0
        for row in csver:
            i += 1
            crsr.execute('''INSERT INTO test VALUES ({}, {}, {}, {}, {})'''.format(int(row[0]), float(row[1]), float(row[2]), int(row[3]), int(row[4])))
            if i % 10000 == 0:
                print(i, "번째 줄 commit")
                conn.commit()

        conn.commit()
        print(i, '개 row commit 완료')
    pass


if __name__ == '__main__':
    projname = os.path.split(sys.path[0])[-1]
    conn = sqlite3.connect(os.path.join(kaggle.path_project(projname), 'test.db'))
    csrr = conn.cursor()

    createtable(conn)

    conn.close()
