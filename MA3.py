""" MA3.py

Student:Felicia Brant-Lundin
Mail: felicia.brant-lundin.8156@student.uu.se
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from numba import njit, prange
import numpy as np

# Exc1
def approximate_pi(n):
    # n is the number of points
    print(f'n = {n}')

    circle = []
    square = []
    n_c = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            n_c += 1
            circle.append((x, y))
        else:
            square.append((x,y))
        
    approx = 4* n_c/n

    x_in, y_in = zip(*circle)
    x_out, y_out = zip(*square)

    plt.figure(figsize=(6, 6))
    plt.title('pi approx')
    plt.axis('equal')
    plt.scatter(x_in, y_in, color='red', s=2)
    plt.scatter(x_out, y_out, color='blue', s=2)
    plt.savefig(f'pi_{n}.png')
    #plt.show()
    plt.close()

    print(f'pi = {approx}')
    return approx

# Exc2, approximation
def sphere_volume(n, d): 
    # n is the number of points
    # d is the number of dimensions of the sphere 
    dots = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
    inne = [p for p in dots if sum(map(lambda x: x**2, p)) <= 1]
    return len(inne) / n * 2**d

#Exc2, real value
def hypersphere_exact(d):
    # n is the number of points
    # d is the number of dimensions of the sphere
    r = 1
    V_d = (m.pi **(d/2)) / (m.gamma((d/2+1))) * r**d
    return V_d

#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points
    # d is the number of dimensions of the sphere
    #np is the number of processes
    
    inne = 0
    for _ in range(n):
        sum_squares = 0
        for _ in range(d):
            x = random.uniform(-1, 1)
            sum_squares += x**2
        if sum_squares <= 1:
            inne += 1

    return (inne/n) * (2**d)

#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes
    n_per_process = n//np
    p = [n_per_process] * np

    with future.ProcessPoolExecutor(max_workers=np) as ex:
        results = list(ex.map(sphere_volume, p, [d]* np))

    return mean(results)

@njit(parallel=True)
def sphere_volume_numba_parallel(n: int, d: int) -> float:
    inne = 0
    for i in prange(n):
        sum_squares = 0.0
        for j in range(d):
            x = random.uniform(-1, 1)
            sum_squares += x**2
        if sum_squares <= 1.0:
            inne += 1

    return (inne / n) * (2**d)
    
def main():
    """# Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(d)}")

    # Exc3
    n = 1000000
    d = 11
    tider = []
    for _ in range(1, 4):
        start = pc()
        sphere_volume(n, d)
        stop = pc()
        tider.append(stop-start)

    tider_n = []
    for _ in range(1, 4):
            start = pc()
            sphere_volume_numba(n, d)
            stop = pc()
            tider_n.append(stop-start)

    print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    print(f'Sequential time (s): {[round(t, 3) for t in tider]}')
    print(f'Sequential time (s): {[round(t, 3) for t in tider_n]}')"""



    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume_numba_parallel(n, d)
    stop = pc()
    tid_seq = stop-start
    print(f"Exc4: Sequential time of {d} and {n}: {tid_seq: .4f} s")

    start = pc()
    sphere_volume_parallel(n, d, np=10)
    stop = pc()
    tid_par = stop - start
    print(f"Exc4: Parallel time (np=10): {tid_par:.4f} s")
    print(f"Speedup: {tid_seq / tid_par:.2f}x")
    

if __name__ == '__main__':
	main()
