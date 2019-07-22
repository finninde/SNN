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


def generate_training_set(zeros, ones, randoms, length_max, seed):
    X = []
    y = []
    np.random.seed(seed)
    while zeros > 0 and ones > 0 and randoms > 0:
        outcome = np.random.randint(0, 3)
        if outcome == 0 and zeros > 0:
            zeros -= 1
            X_add, y_add = sequence_of_zeros(np.random.randint(1, length_max))
            for elem in X_add:
                X.append(elem)
            for elem in y_add:
                y.append(elem)
        elif outcome == 1 and ones > 0:
            X_add, y_add = sequence_of_ones(np.random.randint(1, length_max))
            for elem in X_add:
                X.append(elem)
            for elem in y_add:
                y.append(elem)
            ones -= 1

        elif outcome == 2 and randoms > 0:
            X_add, y_add = random_sequence(np.random.randint(1, length_max))
            for elem in X_add:
                X.append(elem)
            for elem in y_add:
                y.append(elem)
            ones -= 1
            randoms -= 1
    return X, y


def generate_evaluation_set(zeros, ones, randoms, length_max, seed):
    return generate_training_set(zeros, ones, randoms, length_max, seed)