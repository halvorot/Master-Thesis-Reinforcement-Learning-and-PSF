import os
from multiprocessing import Pool
number_of_iter = 1000
cmd = r"python run.py --psf --time=10"
if __name__ == '__main__':
    # start 4 worker processes
    with Pool(processes=6) as pool:
        pool.map(os.system,  [cmd]*number_of_iter)
