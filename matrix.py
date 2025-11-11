import networkx as nx
import matplotlib.pyplot as plt
import copy
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np


MAX_WORKERS = 2
OBJECT_WORKERS = 10

def per_combo_worker(combo_state):
    combo, state = combo_state
    state = list(state)  # convert tuple to mutable list temporarily
    for i in combo:
        if state[i] <= 0:
            return None
        state[i] -= 1
    return tuple(state)  # return as tuple for immutability and efficiency

# Chatgpt
def distribute_identical_objects(num_objects, buckets):
    n = len(buckets)
    results = []

    if num_objects > n:
        return []

    buckets_tuple = tuple(buckets)  # immutable starting state

    # Lazy generator of arguments
    def args_generator():
        for combo in combinations(range(n), num_objects):
            yield (combo, buckets_tuple)

    with ThreadPoolExecutor(max_workers=OBJECT_WORKERS) as executor:
        for r in executor.map(per_combo_worker, args_generator()):
            if r is not None:
                results.append(list(r))

    return results


    # with ThreadPoolExecutor(max_workers=3) as executor:
    # # Submit tasks to the thread pool
    
    # futures = [executor.submit(task, i) for i in range(5)]

    # # Retrieve results as they complete
    # for future in futures:
    #     print(future.result())


def parse_sequence(s: list[str]) -> list[list[int]]:
    results = []
    length = len(s)

    # Start off recursion

    sums = []
    for x in range(0, length):
        sums.append(0)
    
    extra_row = copy.deepcopy(sums)

    def parse_row(iter: int, sequence: list[int], matrix: list[list[int]], sums: list[int]):
        print(iter)
        # Can optimize last iteration since it is deterministic
        # print("")
        # print(f"ITERATION {iter}")
        # print(f"Buckets {sequence}")
        # print(f"Matrix")
        # for x in matrix:
        #     print(x)
        # print("- - - - - - -")
        if iter == length-1: is_last = True; # We want to stop before the last iteration
        else: is_last = False

        # Save value ( amount of edges ) of vertex
        value = sequence[iter]
        sequence[iter] = 0

        # print(f"value {value}")
        # print(f"sequence {sequence}")

        combos = distribute_identical_objects(value, sequence)

        #print(combos)

        def compare_sequences(iter: int, first: list[int], second: list[int]) -> bool:
            a = copy.deepcopy(first)
            b = copy.deepcopy(second)
            for column in range(iter+1, length): # This can be saved instead of compiled everytime
                a[column] = sums[column] + a[column]

            for column in range(iter+1, length):
                b[column] = sums[column] + b[column]

            a.sort() # We need the different amount of each value to be different
            b.sort()
            for i in range(iter+1, length):
                if a[i] != b[i]:
                    return True
            return False
        
        def row_to_matrix_row(iter: int, row: list[int]) -> list[int]:
            matrix = copy.deepcopy(sequence)
            for i in range(iter+1, length):
                matrix[i] = matrix[i] - row[i]
            return matrix

        def test_graphical(iter: int, sequence: list[int]) -> bool:
            non_zero_indexs = 0
            for x in range(iter+1, length):
                if sequence[x] != 0:
                    non_zero_indexs = non_zero_indexs + 1
            if non_zero_indexs == 0: # n=0 is graphical
                return True
            for x in range(iter+1, length):
                if sequence[x] > non_zero_indexs-1:
                    return False
            return True

        unique_rows = []

        # Remove duplicate columns
        for row_num in range(0, len(combos)):
            failed = False
            if len(unique_rows) == 0:
                if test_graphical(iter, combos[row_num]):
                    unique_rows.append(combos[row_num])
            else:
                for unique in unique_rows:
                    if not compare_sequences(iter, combos[row_num], unique) or not test_graphical(iter, combos[row_num]):
                        failed = True
                        break
                if not failed:
                    unique_rows.append(combos[row_num])

        def per_unique_row(row: list[int]):
            m = copy.deepcopy(matrix)
            sums_u = copy.deepcopy(sums)
            m.append(row_to_matrix_row(iter, row))

            if not is_last:
                for column in range(iter+1, length): # This can be saved instead of compiled everytime
                    sums_u[column] = sums_u[column] + row[column]
                parse_row(iter+1, row, m, sums_u)
            else:
                results.append(m)
                return

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for row in unique_rows:
                futures.append(executor.submit(per_unique_row, row))
            for future in as_completed(futures):
                future.result()
        executor.shutdown()

    parse_row(0, s, [], sums)
    return results


s = [5,2,2,2,2,2,5,4,4,4] # 2565
s.reverse()
broken_adjacency_matrix = parse_sequence(s)


# for unique in broken_adjacency_matrix:
#     for row in unique:
#         print(row)
#     print("")

print(len(broken_adjacency_matrix))




# G = nx.from_numpy_array(np.array(broken_adjacency_matrix[1]))
# nx.draw(G, with_labels=True, node_color='skyblue', node_size=1000, font_size=12, font_weight='bold')
# plt.title("Graph Visualization from Adjacency Matrix")
# plt.show()