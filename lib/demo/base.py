import abc
import time
import datetime
import logging

from .utils import load_config

#--------------------------------------------------------------------------------
# base.py contains the demo base class object and its API
#--------------------------------------------------------------------------------


__all__ = ['BaseDemoObject']


class BaseDemoObject(abc.ABC):
    def __init__(self,
            config_path,
            block_length = 15,
            terminal_length = 120):

        # init I/O
        if not isinstance(config_path, str):
            raise TypeError('Argument: config_path must be a string.')

        if not isinstance(block_length, int):
            raise TypeError('Argument: block_length must be a int.')

        if not isinstance(terminal_length, int):
            raise TypeError('Argument: terminal_length must be a int.')

        self.config_path = config_path
        self.config = load_config(config_path)
        self.block_length = block_length
        self.terminal_length = terminal_length
        self.block_num = self.terminal_length // self.block_length

    def demo(self, config = None):
        if config is None:
            config = self.config

        logging.info('Start demo process.')
        start_time = time.time()

        result = self._demo(config)
        self.output_file(result, config)

        total_time = time.time() - start_time
        logging.info('Finish demo process. Total time: ' + str(datetime.timedelta(seconds = total_time)))

        return None

    def _demo(self, config):
        raise NotImplementedError()

    def output_file(self, result, config = None):
        if config is None:
            config = self.config

        self._output_file(result, config)

        return None

    def _output_file(self):
        raise NotImplementedError()


