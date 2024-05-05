import os
from datetime import datetime
import pickle
import numpy as np
import torch


class Logger:

    def __init__(self, dataset, model, *args, **kwargs):

        '''
        Initialize the logging directory:
            ./results/<dataset>/<model>/.../<datetime>/

        Args:
            dataset (str): dataset name.
            model (str): model name.
        '''
        
        self.EXP_DIR = f'./results/{dataset}/{model}'
        self.EXP_DIR += f"/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        self.PICKLE_DIR = f'{self.EXP_DIR}/pickle'
        self.ARRAY_DIR = f'{self.EXP_DIR}/arrays'
        self.TENSOR_DIR = f'{self.EXP_DIR}/tensors'
        os.makedirs(self.EXP_DIR)

    def log(self, text, with_time=True, print_text=False):

        '''
        Write logs to the the logging file: ./<EXP_DIR>/logs

        Args:
            text (str): text to write to the log file.
            with_time (bool): prepend text with datetime of writing.
            print_text (bool): print the text to console, in addition
                to writing it to the log file.
        '''

        if print_text:
            print(text)
        if with_time:
            text = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}: {text}"
        with open(f'{self.EXP_DIR}/logs', 'a') as f:
            f.write(text + '\n')

    def save_pickle(self, fn, obj):

        '''
        Save a Python object as a (binary) pickle file.

        Args:
            fn (str): file name to save the object at.
            obj (Any): Python object to save.
        '''

        if not fn.endswith('.pkl'):
            fn = os.path.splitext(fn)[0] + '.pkl'
        with open(f'{self.PICKLE_DIR}/{fn}', 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_arrays(self, fn, *args, **kwargs):

        '''
        Save NumPy arrays.

        Args:
            args (List[np.ndarray]): arrays saved into <fn>_unnamed.npz 
                and can be queried using integer indexing.
            kwargs (Dict[str, np.ndarray]): arrays saved into <fn>_named.npz 
                and can be queried using string indexing.
        '''

        if args:
            assert all([isinstance(ar, np.ndarray) for ar in args]), \
                f'Expected NumPy arrays, instead received {set((type(ar) for ar in args))}.'
            unnamed_fn = os.path.splitext(fn)[0] + '_unnamed.npz'
            with open(f'{self.ARRAY_DIR}/{unnamed_fn}', 'wb') as f:
                np.savez(f, *args)
        if kwargs:
            assert all([isinstance(ar, np.ndarray) for ar in kwargs.keys()]), \
                f'Expected NumPy arrays, instead received {set((type(ar) for ar in kwargs.keys()))}.'
            named_fn = os.path.splitext(fn)[0] + '_named.npz'
            with open(f'{self.ARRAY_DIR}/{named_fn}', 'wb') as f:
                np.savez(f, **kwargs)

    def save_tensors(self, fn, tensor):

        '''
        Save a PyTorch tensor object.

        Args:
            fn (str): file name to save the tensor at.
            obj (torch.Tensor): Torch tensor to save.
        '''

        assert isinstance(tensor, torch.Tensor), \
            f'Expected Torch tensor, instead received {type(tensor)}.'
        if not fn.endswith('.pt'):
            fn = os.path.splitext(fn)[0] + '.pt'
        torch.save(tensor, f'{self.TENSOR_DIR}/fn')