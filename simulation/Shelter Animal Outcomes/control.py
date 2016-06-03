def targetdata(dataset: list, attribute=''):
    """

    :param dataset: header가 포함된 dataset
    :param attribute: target으로 분류할 attribute
    :return:
    """
    headers = dataset.pop(0)
    attridx = 0
    if isinstance(headers, list):
        attridx = headers.index(attribute)

    target, trdata = [], []
    for i, col in enumerate(zip(*dataset)):
        if i == attridx:
            target = col
        else:
            trdata.append(col)

    data = tuple(zip(*trdata))

    return target, data


def feature_average(feature_of_classes: dict, average=1):
    """
    각 class의 feature들의 x를 구해서 각 x에 대한 평균을 구합니다.
    :param feature_of_classes:
    :param average: 평균을 낼 X의 개수
    :return:
    """
    randvarrange = list(set([set(feature_of_classes[x]) for x in feature_of_classes])).sort()

    return 1


def feature_probability_estimation(feature_of_classes: dict):
    """
    각 class의 feature들의 x를 구해서 각 x에 대한 probability estimation을 합니다.
    :param feature_of_classes:
    :return:
    """
    return 1
