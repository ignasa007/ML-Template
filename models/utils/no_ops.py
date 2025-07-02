class NoOp:
    def __init__(self, *args, **kwargs):
        ...
    def __call__(self, input):
        return input
    
def no_op(input):
    return input