import abc

#--------------------------------------------------------------------------------
# base.py contains the meta class of all transform function
#--------------------------------------------------------------------------------

__all__ = ['TransformFunctionObject']


class TransformFunctionObject(abc.ABC):
    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError()

 
