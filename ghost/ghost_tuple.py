"""A useful type for dealing with ghost versions of objects"""
import operator


class GhostTuple(object):
    """GhostTuple helps with working with any two objects with the same structure.

    The two objects are stored in `live` and `ghost`.

    When iterating over a GhostTuple, the live and ghost values are iterated over
    in parallel, and each value is returned as a GhostTuple.

    When accessing a GhostTuple, the live and ghost values are accessed in parallel,
    and the value is returned as a GhostTuple.

    Example:
    >>> a = GhostTuple(live=[1, 2, 3], ghost=[4, 5, 6])
    >>> for b in a:
    ...     print(b.live, b.ghost) # Works!
    ...
    1 4
    2 5
    3 6

    >>> b = a[0]
    >>> print(b.live, b.ghost)
    1 4

    >>> a = GhostTuple(live={"k": 1}, ghost={"k": 2})
    >>> b = a["k"]
    >>> print(b.live, b.ghost)
    1 2

    >>> a = GhostTuple(live=1, ghost=2)
    >>> b = a + a
    >>> c = 2 + a
    >>> print(b.live, b.ghost)
    2 4
    >>> print(c.live, c.ghost)
    3 4

    >>> f = GhostTuple(live=lambda x: 2*x, ghost=lambda x: 3*x)
    >>> print(f(1).live, f(1).ghost)
    2 3
    """

    def __init__(self, live, ghost):
        """Set live and ghost attributes."""
        self.live = live
        self.ghost = ghost

    def __iter__(self):
        """Loop through ghost and live."""
        for (a_i, b_i) in zip(self.live, self.ghost):
            yield GhostTuple(live=a_i, ghost=b_i)

    def __getitem__(self, index):
        """Get the live or ghost value."""
        if isinstance(index, GhostTuple):
            return GhostTuple(
                live=self.live[index.live],
                ghost=self.ghost[index.ghost],
            )
        else:
            return GhostTuple(live=self.live[index], ghost=self.ghost[index])

    def __getattr__(self, name):
        """Get the live or ghost value."""
        if isinstance(name, GhostTuple):
            return GhostTuple(
                live=getattr(self.live, name.live), 
                ghost=getattr(self.ghost, name.ghost),
            )
        else:
            return GhostTuple(
                live=getattr(self.live, name),
                ghost=getattr(self.ghost, name),
            )

    def __neg__(self):
        """Negate the live and ghost values."""
        return GhostTuple(
            live=-self.live,
            ghost=-self.ghost,
        )

    def __call__(self, *args, **kwargs):
        """Call the live and ghost values."""
        return GhostTuple(
            live=self.live(*args, **kwargs),
            ghost=self.ghost(*args, **kwargs),
        )


# Define all operators for GhostTuple.
#  This is equivalent to defining
# __add__(self, other)
# for all operators.
def apply_op_to_ghost(func, reversed=False):
    if reversed:

        def inner(self, other):
            if isinstance(other, GhostTuple):
                return GhostTuple(
                    live=func(other.live, self.live),
                    ghost=func(other.ghost, self.ghost),
                )
            else:
                return GhostTuple(
                    live=func(other, self.live),
                    ghost=func(other, self.ghost),
                )

    else:

        def inner(self, other):
            if isinstance(other, GhostTuple):
                return GhostTuple(
                    live=func(self.live, other.live),
                    ghost=func(self.ghost, other.ghost),
                )
            else:
                return GhostTuple(
                    live=func(self.live, other),
                    ghost=func(self.ghost, other),
                )

    return inner


# fmt: off
all_operators = (
    "add sub mul mod matmul".split(" ")
    + "truediv floordiv or pow".split(" ")
)
all_operators += ["r" + op for op in all_operators]
# fmt: on

for op in all_operators:

    reverse = op.startswith("r")
    op_name = "__{}__".format(op)
    op_name_in_operator_module = op_name

    if reverse:
        op_name_in_operator_module = "__{}__".format(op[1:])

    setattr(
        GhostTuple,
        op_name,
        apply_op_to_ghost(getattr(operator, op_name_in_operator_module), reverse),
    )
