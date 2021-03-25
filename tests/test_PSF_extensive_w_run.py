
import os
from multiprocessing import Pool
from pathlib import Path

number_of_iter = 1000
time = 1000

HERE = Path(__file__).parent
os.chdir(HERE.parent)

cmd = f"python run.py --psf --time={time} --env=CrazyAgent-v0"
if __name__ == '__main__':
    # start 6 worker processes

    print(f"Running {number_of_iter} number of iterations, each {time} s through 'run.py'")
    number_of_workers = 6
    print(f"Spinning up {number_of_workers} processes")
    with Pool(processes=6) as pool:
        pool.map(os.system,  [cmd]*number_of_iter)
