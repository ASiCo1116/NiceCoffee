import abc

#--------------------------------------------------------------------------------
# base.py contains the meta class of all evaluation methods.
#--------------------------------------------------------------------------------

class Report(abc.ABC):
    @abc.abstractmethod
    def summary(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self):
        raise NotImplementedError()


