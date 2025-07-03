from typing import Any, List, Optional


# Copied from strtobool in distutils/util.py (deprecated Python 3.12 onwards)
def strtobool(val):
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("Invalid truth value %r" % (val,))


def interpret_args(args: str, to: type, default: Any, num_layers: Optional[int] = None) -> List:
    """
    Parses args in format `type type*int type*int...`
    e.g. say `to` is `int`, then `args = "64 32*3 16*2"` is split as
        `out = [64, 32, 32, 32, 16, 16]`.
    e.g. say `to` is `bool`, then `args = "True*3 f*2 0"` is split as
        `out = [True, True, True, False, False False]`; see function `strtobool`.

    If args is split into one value only, then repeat it for `num_layers` number of times.
    e.g. `args=16, to=int, num_layers=3` is split as 
        `out = [16, 16, 16]`.
    """

    if to == bool:
        to = strtobool
    def cast(arg: str):
        if arg.lower() in ('none', 'null'):
            return None
        else:
            return to(arg)

    args = str(args)
    out = list()
    for arg in args.split():
        if not arg:
            continue
        elif "*" in arg:
            size, mult = arg.split("*")
            out.extend([cast(size)]*int(mult))
        else:
            out.append(cast(arg))
    
    if isinstance(num_layers, int) and num_layers < len(out):
        raise RuntimeError(f"Parsed more args than number of layers: {args = }, {num_layers=}, {out = }.")
    elif isinstance(num_layers, int) and num_layers > len(out):
        if len(out) == 1:
            out = out * num_layers
        else:
            raise RuntimeError(f"Cannot handle `1 < len(out) < num_layers`: {args = }, {num_layers=}, {out = }.")

    return out


def make_kwargs(**kwargs):
    out = kwargs.copy()
    for k, v in kwargs.items():
        if v is None:
            del out[k]
    return out