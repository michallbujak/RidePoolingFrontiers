from numba import jit

@jit
def count_zeros(
        some_list
):
    counter = 0
    for t in some_list:
        if t[0] == 0:
            counter +=1
    return counter


some_list = [0, 0]*10

print(count_zeros(some_list))
