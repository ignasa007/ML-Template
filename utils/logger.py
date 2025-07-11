from typing import Any, Union, List, Tuple, Dict
from argparse import Namespace
import os
from datetime import datetime
import pickle

import yaml
from yacs.config import CfgNode
import numpy as np
import torch
from torch import Tensor


class Logger:

    @staticmethod
    def timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    @staticmethod
    def process_kwargs(kwargs: Dict) -> Dict[str, Any]:
        # Expecting kwargs["kwargs"] to be a dictionary in itself
        #   => update kwargs with kwargs["kwargs"]
        if "kwargs" in kwargs:
            kwargs.update(kwargs.pop("kwargs"))
        return kwargs

    @staticmethod
    def sci_notation(x: Union[int, float], decimals: int = 6, strip: bool = True) -> str:
        mantissa, exponent = f"{x:.{decimals}e}".split("e")
        if strip:
            mantissa = mantissa.rstrip("0").rstrip(".")
        return mantissa + f"e{exponent}"

    @staticmethod
    def fix_ext(fn: str, default_ext: str, force_ext: bool = False):
        assert len(default_ext) > 1
        root, ext = os.path.splitext(fn)
        # `fn` already has a valid extension and `force_ext` is False
        if len(ext) > 1 and not force_ext:
            return fn
        # Add extension if `fn` has an invalid extension or `force_ext` is True
        else:
            return root + default_ext

    def __init__(self, args: Namespace, cfg: CfgNode):
        """
        Initialize the logging directory.
        Args:
            args (argparse.Namespace): command-line arguments.
            cfg (yacs.CfgNode): configurations picked from ./config/.
        """

        self.EXP_DIR = cfg.exp_dir
        self.OBJECTS_DIR = f"{self.EXP_DIR}/objects"
        os.makedirs(self.OBJECTS_DIR)
        self.save_objects(args=args, cfg=cfg)

        # Log the command-line arguments
        self.log(yaml.safe_dump(vars(args), indent=4), with_time=False)
        # Log the rest of the configurations
        self.log(cfg.dump(indent=4), with_time=False)

    def log(self, text: str, with_time: bool = True, print_text: bool = False) -> None:
        """
        Write logs to the the logging file: ./<EXP_DIR>/logs
        Args:
            text (str): text to write to the log file.
            with_time (bool): prepend text with datetime of writing.
            print_text (bool): print the text to console, in addition
                to writing it to the log file.
        """

        if print_text:
            print(text)
        if with_time:
            text = f"[{Logger.timestamp()}] {text}"
        with open(f"{self.EXP_DIR}/logs", "a") as f:
            f.write(text + "\n")

    def log_metrics(
        self,
        metrics: List[Tuple[str, Tensor]],
        prefix: str = "",
        with_time: bool = True,
        print_text: bool = False
    ) -> None:

        formatted_metrics = prefix
        formatted_metrics += ", ".join(
            f"{name} = {Logger.sci_notation(value.item(), decimals=6, strip=False)}"
            for name, value in metrics
        )
        self.log(formatted_metrics, with_time, print_text)

    def save_objects(self, **kwargs) -> None:
        """
        Save Python objects as (binary) pickle files.
        Args:
            kwargs (Dict[str, Object]): <value> to be saved into <key>.pickle
        Args can be passed as
            - `Logger.save_objects(a=1, b=2, c=3, d=4)`
            - `Logger.save_objects(kwargs={"a": 1, "b": 2}, c=3, d=4)`
            - `Logger.save_objects(kwargs={"a": 1, "b": 2, "c": 3, "d": 4})`
        In each case, the kwargs will be transformed as
            `kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}`
        """

        kwargs = Logger.process_kwargs(kwargs)
        for fn, obj in kwargs.items():
            fn = Logger.fix_ext(fn, default_ext=".pickle")
            with open(f"{self.OBJECTS_DIR}/{fn}", "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_arrays(self, **kwargs) -> None:
        """Save NumPy arrays."""
        kwargs = Logger.process_kwargs(kwargs)
        for fn, arr in kwargs.items():
            fn = Logger.fix_ext(fn, default_ext=".npy", force_ext=True)
            np.save(file=f"{self.OBJECTS_DIR}/{fn}", arr=arr, allow_pickle=True)

    def save_tensors(self, **kwargs) -> None:
        """Save PyTorch tensors. Your responsibility to detach from the computational graph."""
        kwargs = Logger.process_kwargs(kwargs)
        for fn, tensor in kwargs.items():
            fn = Logger.fix_ext(fn, default_ext=".pt", force_ext=True)
            torch.save(tensor, f=f"{self.OBJECTS_DIR}/{fn}")