import bisect
import random
import scipy.stats as ss

z = random.random()
probs = [0.1, 0.5, 1]
args = (([1, 2], [5, 5]), ([3, 4], [5, 5]), ([5, 6], [5, 5]))

index = bisect.bisect(probs, z)


def mixed_discrete_norm_distribution(probs, *args):
    z = random.random()
    index = bisect.bisect(probs, z)
    a1 = args[index][0]
    a2 = args[index][1]

    def internal_function(*X):
        a3 = list(zip(X, args[index][0], args[index][1]))
        for x, mean, std in a3:
            r = ss.norm.ppf(x, loc=mean, scale=std)

        return [ss.norm.ppf(x, loc=mean, scale=std) for x, mean, std in zip(X, args[index][0], args[index][1])]

    return internal_function


print(mixed_discrete_norm_distribution(probs, *args)(0.1, 0.2))

