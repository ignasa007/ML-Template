import os
from datetime import datetime
import pickle
import numpy as np
import torch


class Logger:

    def __init__(self, dataset, model, *args, **kwargs):
        self.EXP_DIR = f'./results/{dataset}/{model}'
        self.EXP_DIR += f"/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        os.makedirs(self.EXP_DIR)

    def log(self, text, with_time=True, print_text=False):
        if print_text:
            print(text)
        if with_time:
            text = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}: {text}"
        with open(f'{self.EXP_DIR}/logs', 'a') as f:
            f.write(text + '\n')

    def save_pickle(self, fn, obj):
        if not fn.endswith('.pkl'):
            fn = os.path.splitext(fn)[0] + '.pkl'
        with open(f'{self.EXP_DIR}/{fn}', 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_arrays(self, fn, *args, **kwargs):
        if args:
            assert all([isinstance(ar, np.ndarray) for ar in args]), \
                f'Expected NumPy arrays, instead received {set((type(ar) for ar in args))}.'
            unnamed_fn = os.path.splitext(fn)[0] + '_unnamed.npz'
            with open(f'{self.EXP_DIR}/{unnamed_fn}', 'wb') as f:
                np.savez(f, *args)
        if kwargs:
            assert all([isinstance(ar, np.ndarray) for ar in kwargs.keys()]), \
                f'Expected NumPy arrays, instead received {set((type(ar) for ar in kwargs.keys()))}.'
            named_fn = os.path.splitext(fn)[0] + '_named.npz'
            with open(f'{self.EXP_DIR}/{named_fn}', 'wb') as f:
                np.savez(f, **kwargs)

    def save_tensors(self, fn, tensor):
        assert isinstance(tensor, torch.Tensor), \
            f'Expected Torch tensor, instead received {type(tensor)}.'
        if not fn.endswith('.pt'):
            fn = os.path.splitext(fn)[0] + '.pt'
        torch.save(tensor, fn)