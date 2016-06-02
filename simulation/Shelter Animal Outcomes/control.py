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
