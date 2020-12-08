from ..utils import load_config

#--------------------------------------------------------------------------------
# utils.py contains the function and class object used in lib.plot module
#--------------------------------------------------------------------------------


__all__ = ['load_config']

def init_matplotlib_without_gui():
    import matplotlib
    matplotlib.use('agg')

    return None


