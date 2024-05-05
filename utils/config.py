from typing import Union
from yacs.config import CfgNode as CN


def default_cfg():

    '''
    The default configuration object for the experiments.
        - Need to register all configurations that are expected.

    Return:
        _C: A confiduration object with placeholder values.
    '''

    _C = CN()
    # _C.property = None

    _C.DATA = CN() # base dataset parameters
    # _C.DATA.property = None
    _C.DATA.args = CN() # run-specific dataset parameters
    # _C.DATA.args.property = None

    _C.MODEL = CN() # base model parameters
    # _C.MODEL.property = None
    _C.MODEL.args = CN() # run-specific model parameters
    # _C.MODEL.args.property = None

    return _C.clone()


class Config:
    
    def __init__(self, root, dataset, model, override=None):

        '''
        Initialization of the configuration object used by the main file.

        Args:
            root (str): file path for the default configurations.
            dataset (str): file path for the dataset configurations used in
                the experiment.
            model (str): file path for the model configurations used in the
                experiment.
            override (Union[list, None]): key-value pairs with command-line
                arguments indicating the configurations to override.
        '''

        self.cfg = default_cfg()
        self.cfg.merge_from_file(f'{root}/config.yaml')
        self.cfg.DATA.merge_from_file(f'{root}/datasets/{dataset}.yaml')
        self.cfg.MODEL.merge_from_file(f'{root}/models/{model}.yaml')

        if isinstance(override, list):
            self.cfg.merge_from_list(override)

    def __getattr__(self, name: str):

        '''
        Method for returning configurations using dot operator.
        '''

        return self.cfg.__getattr__(name)