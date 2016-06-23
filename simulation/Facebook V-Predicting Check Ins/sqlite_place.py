import os
import sys
import sqlite3
import csv
sys.path.append(os.path.abspath('../'))
import kaggle


def createtable(trainconn, placeconn):
    tcsrr, pcsrr = trainconn.cursor(), placeconn.cursor()
    tcsrr.execute('''SELECT DISTINCT place_id from train''')
    pidlist = tuple([x[0] for x in tcsrr])
    pcsrr.execute('''ATTACH DATABASE \'{}\' AS tr'''.format(os.path.join(kaggle.path_project(projname), 'train.db')))
    print('총', len(pidlist), '개의 place id확인, 이를 Class로 규정')

    # pcsrr.executemany('''CREATE TABLE ? AS SELECT x, y, accuracy, time FROM tr.train WHERE place_id=?''', (tuple(['C' + a for a in pidlist]), pidlist))
    # table 이름에는 ? expression을 쓸 수 없음
    i = 0
    for pid in pidlist:
        pcsrr.execute('''CREATE TABLE {} AS SELECT x, y, accuracy, time FROM tr.train WHERE place_id={}'''.format('C' + pid, pid))
        i += 1
        # print('C' + pid, end=' ',flush=True)
        if i % 1000 == 0:
            placeconn.commit()
            print(i, '번째 create table commit')
    placeconn.commit()
    print('총', i, '개의 create table commit complete')

    # piddict, i = {x: 0 for x in pidlist}, 0
    # for output in tcsrr:
    #     placeid, x, y, acc, time = output
    #     if placeid not in pidlist:
    #         print(placeid, ' does not exist. Wrong initialization.')
    #         break
    #     piddict[placeid] += 1
    #     i += 1
    #     pcsrr.execute('''CREATE TABLE IF NOT EXISTS p{} (x REAL, y REAL, accuracy INTEGER, time INTEGER)'''.format(placeid))
        # pcsrr.execute('''INSERT INTO p{} VALUES ({}, {}, {}, {})'''.format(placeid, x, y, acc, time))
        # i += 1
        # if i % 10000 == 0:
        #     print(i, "번째 줄 commit")
        #     placeconn.commit()
    # placeconn.commit()
    #
    # print(i, '개 sample commit 완료')
    # for pid in piddict:
    #     print(pid, ' : ', piddict[pid])


if __name__ == '__main__':
    projname = os.path.split(sys.path[0])[-1]
    conn_train = sqlite3.connect(os.path.join(kaggle.path_project(projname), 'train.db'))
    conn_place = sqlite3.connect(os.path.join(kaggle.path_project(projname), 'place.db'))

    createtable(conn_train, conn_place)

    conn_train.close()
    conn_place.close()
