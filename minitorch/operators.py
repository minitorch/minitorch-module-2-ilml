"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Iterator

from typing import TypeVar

A = TypeVar('A')
T = TypeVar('T')
U = TypeVar('U')

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:
    return x * y


def id(x: A) -> A:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    return 1 / (1 + math.e ** (-x))


def relu(x: float) -> float:
    return max(0.0, x) 


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    if x == 0:
        ValueError("Cannot take reciprocal of zero.")
    return 1 / x


def log_back(x: float, d_out: float) -> float:
    return inv(x) * d_out


def inv_back(x: float, d_out: float) -> float:
    return (-1 / x**2) * d_out


def relu_back(x: float, d_out: float) -> float:
    return d_out if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[T], U], iter: Iterable[T]) -> Iterator[U]:
    for it in iter:
        yield fn(it)

    
def zipWith(fn: Callable[[A, T], U], iter1: Iterable[A], iter2: Iterable[T]) -> Iterator[U]:
    for x, y in zip(iter1, iter2):
        yield fn(x, y)


def reduce(fn: Callable[[U, U], U], seq: Iterable[U], initial: U = None) -> U:
    it = iter(seq)
    
    if initial is None:
        try:
            acc = next(it)
        except StopIteration:
            raise ValueError("reduce() of empty sequence with no initial value")
    else:
        acc = initial
    
    for item in it:
        acc = fn(acc, item)
    return acc


def negList(seq: list[float]) -> list[float]:
    return list(map(neg, seq))


def addLists(seq1: list[float], seq2: list[float]) -> list[float]:
    return list(zipWith(add, seq1, seq2))


def sum(seq: list[float]) -> float:
    if not seq:  # Handle empty list
        return 0.0
    return reduce(add, seq)


def prod(seq: list[float]) -> float:
    if not seq:  # Handle empty list
        return 1.0
    return reduce(mul, seq)