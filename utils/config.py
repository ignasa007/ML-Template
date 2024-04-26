from typing import Union
from yacs.config import CfgNode as CN


def default_cfg():

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
    
    def __init__(self, root: str, dataset: str, model: str, override: Union[list, None] = None):

        self.cfg = default_cfg()
        self.cfg.merge_from_file(f'{root}/config.yaml')
        self.cfg.DATA.merge_from_file(f'{root}/datasets/{dataset}.yaml')
        self.cfg.MODEL.merge_from_file(f'{root}/models/{model}.yaml')

        if isinstance(override, list):
            self.cfg.merge_from_list(override)

    def __getattr__(self, name: str):

        return self.cfg.__getattr__(name)