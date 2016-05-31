"""
각종 데이터를 시각화합니다
"""
from matplotlib import pyplot


def plotdots(*args, **kwargs):
    pass


def plotlines(*args, **kwargs):
    pass


class InputParser:
    def __init__(self):
        self._value = ''

    def __add__(self, other):
        self._value = other

    def __str__(self):
        return self._value

    def __call__(self, *args, **kwargs):
        self._value = input()
        return self._value


if __name__ == '__main__':
    parser = InputParser()
    whosay = ""
    while parser():
        print(parser)
    pass