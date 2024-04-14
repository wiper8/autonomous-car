import math

def sign(x):
    if x>0:
        return 1
    if x<0:
        return -1
    else:
        return 0

def seq(from_, to, step):
    assert to >= from_
    sequence = []
    sequence.append(from_)

    while sequence[-1] + step <= to:
        sequence.append(sequence[-1] + step)

    return sequence