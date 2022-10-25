import random
import multiprocessing as mp


def foo():
    return random.random()


def init_pool_processes():
    random.seed()


def foo2():
    pool = mp.pool.ThreadPool(mp.cpu_count(), initializer=init_pool_processes)
    res = pool.map(foo, range(3))
    pool.close()
    return res
