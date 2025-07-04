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


def interpret_args(args: str, to: type, num_layers: Optional[int] = None) -> List:
    
    """
    Parses args in format `type type*int type*int...`
    e.g. say `to` is `int`, then `args = "64 32*3 16*2"` is split as
        `out = [64, 32, 32, 32, 16, 16]`.
    e.g. say `to` is `bool`, then `args = "True*3 f*2 0"` is split as
        `out = [True, True, True, False, False False]`; see function `strtobool`.

    `None`, strings `"None"` and `"Null"`, and their lower-case versions are parsed as None.
    Combinations of the characters `"."`, `"-"`, and `"_"` are parsed as Ellipsis;
        to be left out when making kwargs (see `make_kwargs` below).
    If args is split into exactly one value, copy it `num_layers` times over.
    """

    if to == bool:
        to = strtobool

    def cast(arg: str):
        if arg.lower() in ('none', 'null'):
            return None
        elif set(arg).issubset(('.', '-', '_')):
            return ...
        return to(arg)

    args = str(args)
    out = list()
    for arg in args.split():
        if arg == "":
            continue
        elif "*" in arg:
            value, multiplicity = arg.split("*")
            out.extend([cast(value)]*int(multiplicity))
        else:
            out.append(cast(arg))
    
    if isinstance(num_layers, int):
        if len(out) == 1:
            out = out * num_layers
        elif len(out) != num_layers:
            raise RuntimeError(
                "Number of args parsed is not equal to the number of layers:"
                f" args = {args}, to = {to.__name__}, num_layers = {num_layers} -> out = {out}."
            )

    return out


def make_kwargs(**kwargs):
    """Ellipsis are not included in kwargs."""
    out = kwargs.copy()
    for k, v in kwargs.items():
        if v == Ellipsis:
            del out[k]
    return out