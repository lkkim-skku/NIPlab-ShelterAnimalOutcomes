def histogram(x: list, bins: int):
    """
    히스토그램
    :param x: array-like
    :param bins: bin 갯수
    :return: histogram: array-like.
    :return: steps: array-like. 각 bin의 경계값. 만약 어떤 bin이 [0, 0.5, 1]이라면 step은 [0.5, 1]임.
    """

    x = [a for a in x]
    x.sort()
    xn, xx = x[0], x[-1]
    step = (xx - xn) / bins
    box, i = xn + step, 0
    histo = []
    for a in x:
        while a > box:
            histo.append(i)
            i = 0
            box += step
        i += 1
    if len(histo) == bins:
        histo[-1] += i
    else:
        # histo.append(i)
        while len(histo) < bins:
            histo.append(i)
            i = 0
    steps = []
    box = xn + step
    for _ in range(bins):
        steps.append(box)
        box += step
    return histo, steps


def histo_cudif(x: list, bins: int):
    """
    histogram Cumulative Distribution Function

    :param x:
    :param bins:
    :return: array-like.
    """
    hist, steps = histogram(x, bins)

    k = 0.
    cdf_ = []
    samplan = len(x)
    for h in hist:
        k += h / samplan
        cdf_.append(k)

    return cdf_