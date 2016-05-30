from simulation import kaggleio


def sexuponoutcome_160530():
    pass


def outcomesubtype_160530():
    pass


def breed_160530():
    pass


def colour_160530():
    pass


class SAO160530(kaggleio.DataSet):
    def __init__(self):
        super().__init__()
        self._set = dict()
        self.callbacks = dict()

    def fit(self, x):
        super().fit(x)
        for head in self.header:
            if head == 'OutcomeType':
                pass
    pass
