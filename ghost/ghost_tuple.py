
import operator

class GhostTuple(object):

    def __init__(self, live, ghost):
        
        self.live = live
        self.ghost = ghost

    def __iter__(self):

        for (a, b) in zip(self.live, self.ghost):
            yield GhostTuple(a, b)

    def __getitem__(self, index):

        if isinstance(index, GhostTuple):
            return GhostTuple(self.live[index.live], self.ghost[index.ghost])
        else:
            return GhostTuple(self.live[index], self.ghost[index])
        
    def __getattr__(self, name):

        if isinstance(name, GhostTuple):
            return GhostTuple(getattr(self.live, name.live), getattr(self.ghost, name.ghost))
        else:
            return GhostTuple(getattr(self.live, name), getattr(self.ghost, name))

    def __neg__(self):

        return GhostTuple(-self.live, -self.ghost)

    def __call__(self, *args, **kwargs):
        
        return GhostTuple(self.live(*args, **kwargs), self.ghost(*args, **kwargs))


def apply_op_to_ghost(func, reverse=False):
    
    if reverse:
        def inner(self, other):
            if isinstance(other, GhostTuple):
                return GhostTuple(func(other.live, self.live), func(other.ghost, self.ghost))
            else:
                return GhostTuple(func(other, self.live), func(other, self.ghost))

    else:
        def inner(self, other):
            if isinstance(other, GhostTuple):
                return GhostTuple(func(self.live, other.live), func(self.ghost, other.ghost))
            else:
                return GhostTuple(func(self.live, other), func(self.ghost, other))

    return inner
    
all_operators = "add sub mul mod matmul truediv floordiv or pow".split(' ')

for op in all_operators:
    
    # normal
    op_name = "__{}__".format(op)
    op_name_in_operator_module = op_name

    setattr(GhostTuple, op_name, apply_op_to_ghost(getattr(operator, op_name_in_operator_module)))

    # reverse
    op_name = "__{}__".format("r" + op)
    setattr(GhostTuple, op_name, apply_op_to_ghost(getattr(operator, op_name_in_operator_module), reverse=True))