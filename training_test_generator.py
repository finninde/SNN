import nuympy as np

def sequence_of_ones(length):
    # First Reset
    X = [0]
    # Then add input
    for elem in [1]*length:
        X.append(elem)
    # First reset is defined as not pair
    y = [0]
    if length % 2 == 0:
        for elem in [0,1]*(length//2):
            y.append(elem)
    else:
        for elem in [0,1]*((length-1)//2):
            y.append(elem)
        y.append(0)
    return X, y

def sequence_of_zeros(length):
    # Reset
    X = [0]
    y= [0]
    for elem in [0]*length:
        X.append(elem)
    for elem in [0]*(length):
        y.append(elem)
    return X,y


def random_sequence(length, seed):
    # reset
    X = [0]
    y = [0]
    np.random.seed(seed)
    for i in range(0, length):
        X.append(np.random.randint(0, 2))

        if X[-1] == 1 and X[-2] == 1:
            y.append(1)
        else:
            y.append(0)
    return X, y