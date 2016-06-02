"""
controller
"""


def mymax(itervar, **kwargs):
    """
    max인데, max의 결과가 일정 기준 이하면 default를 리턴
    :param dict itervar:
    :param kwargs:
    :return:
    """
    if 'key' in kwargs:
        mval = max(itervar, key=kwargs['key'])
    else:
        mval = max(itervar)
    default = kwargs['default'] if 'default' in kwargs else ValueError
    if 'underbound' in kwargs:
        mval = default if itervar[mval] < kwargs['underbound'] else mval
    return mval


def mymin(itervar, **kwargs):
    """
    max인데, max의 결과가 일정 기준 이하면 default를 리턴
    :param dict itervar:
    :param kwargs:
    :return:
    """
    if 'key' in kwargs:
        mval = min(itervar, key=kwargs['key'])
    else:
        mval = min(itervar)
    default = kwargs['default'] if 'default' in kwargs else ValueError
    if 'underbound' in kwargs:
        mval = default if itervar[mval] > kwargs['upperbound'] else mval
    return mval