from typing import Optional, Union, Tuple, List
import itertools

import torch


def interpret_size_as_int(size, total):
    if size is None:
        return None
    elif isinstance(size, float) and 0. <= size <= 1.:
        return int(size*total)
    elif isinstance(size, int) and size >= 0:
        return size
    else:
        raise ValueError(f"Received invalid size: `{size=}`, `type(size) = {type(size).__name__}`.")

def get_split_indices(
    main_size: Union[float, int, None],
    other_sizes: Optional[List[Union[float, int, None]]],
    total_size: int
) -> Tuple[torch.Tensor, List[torch.Tensor]]:

    """
    Get random indices for splitting datasets.
    Args:
        main_size (float/int/None): the primary split (e.g. train) - it is treated differently.
        other_sizes (List[float/int/None]): other splits (e.g. validation).
        total_size (int): total size to split into (e.g. len(dataset)).
    Returns:
        main_indices (torch.Tensor): indices corresponding to `main_size`.
        other_indices (List[torch.Tensor]): indices corresponding to `other_sizes`.

    Treating None:
        - If any of `other_sizes` is None, it is treated as 0.
        - If `main_size` is None, it is assigned the remaining samples.
    Treating integer sizes: any integer size is used exactly
    Treating float sizes: any float size is treated as a fraction of `total_size`.
    If the sizes are float or None (-> 0.), and sum close to 1, then
        - assign any remaining samples in a round-robin fashion,
        - starting with `main_size`.
    """

    NoneType = type(None)

    if other_sizes is None:
        other_sizes = list()
    original_sizes = [main_size] + other_sizes

    if not (
        all((isinstance(size, (int, NoneType)) for size in original_sizes)) or
        all((isinstance(size, (float, NoneType)) for size in original_sizes))
    ):
        raise TypeError(f"Received mixed types: `type(main_size) = {type(main_size).__name__}`, " \
            f"`type(other_sizes) = ({', '.join((type(size).__name__ for size in other_sizes))})`.")

    for i, size in enumerate(other_sizes):
        # In case size is passed but as integer 0, it will fail type check below
        if size is None:
            other_sizes[i] = 0

    # INTERPRET SIZES AS INTEGERS

    all_sizes = list(map(lambda size: interpret_size_as_int(size, total_size), [main_size]+other_sizes))
    if not (all_sizes[0] is None or 0. < all_sizes[0] <= total_size):
        raise ValueError(
            f"`main_size` must be in `(0, len(dataset)]`. Instead interpreted as `{all_sizes[0]}`." \
            f" Derived from `main_size={original_sizes[0]}` and `total_size={total_size}`."
        )
    for i in range(1, len(all_sizes)):
        # NoneType `other_size` replaced with 0 above
        if not (0. <= all_sizes[i] < total_size):
            raise ValueError(
                f"`other_size` must be in `[0, len(dataset))`. Instead interpreted as `{all_sizes[i]}`." \
                f" Derived from `other_sizes[i]={original_sizes[i]}` and `total_size={total_size}`."
            )

    # Assign the rest to training
    if all_sizes[0] is None:
        all_sizes[0] = total_size - sum(other_sizes)
    
    # Check total size is not more than the length of the dataset
    remainder = total_size - sum(all_sizes)
    assert remainder >= 0, \
        f"`sum(all_sizes) ({sum(all_sizes)}) > total_size ({total_size})`. Derived from" \
        f" `main_size={original_sizes[0]}`, `other_sizes = {original_sizes[1:]}` and `total_size={total_size}`."
    # Assign remainder only if all sizes requested are fractions and almost sum to 1
    # Absolute tolerance = 1%, e.g. reassigns for 0.33 and 0.66, but not 0.3 and 0.6
    if (
        all((isinstance(size, (float, NoneType)) for size in original_sizes)) and
        abs(1 - sum((size if size is not None else 0. for size in original_sizes))) <= 1e-2
    ):
        pointer = 0
        while remainder > 0:
            # Assign only to sets which had non-zero float sizes
            if isinstance(original_sizes[pointer], float) and original_sizes[pointer] > 0.:
                all_sizes[pointer] += 1
                remainder -= 1
            pointer = pointer+1 if pointer < len(all_sizes)-1 else 0

    # GET RANDOM INDICES

    all_indices = torch.randperm(total_size)
    main_indices, *other_indices = [
        all_indices[offset-length:offset]
        for offset, length in zip(itertools.accumulate(all_sizes), all_sizes)
    ]

    return main_indices, other_indices