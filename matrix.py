from typing import Iterable, List, TypeVar
import networkx as nx
import matplotlib.pyplot as plt
import copy
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import time
from itertools import chain
import gc


MAX_TOP_WORKERS = 20
MAX_SUB_WORKERS = 5

def distribute_identical_objects(num_objects, buckets):
    """
    Generate all valid distributions of `num_objects` identical objects into `buckets`,
    where each bucket has an initial capacity given in `buckets`.

    Returns a list of tuples representing remaining bucket states after distribution.
    """
    n = len(buckets)
    if num_objects > n:
        return []

    buckets_tuple = tuple(buckets)
    results = []

    for combo in combinations(range(n), num_objects):
        # Check if all indices in combo have enough capacity
        valid = True
        for idx in combo:
            if buckets_tuple[idx] <= 0:
                valid = False
                break
        if not valid:
            continue

        # Apply decrement to a copy of the state
        new_state = list(buckets_tuple)
        for idx in combo:
            new_state[idx] -= 1

        results.append(list(new_state))

    return results

# Chatgpt
T = TypeVar("Matrix")  # Generic type for items

def remove_duplicates_recursive_gen_eq(
    data_gen: Iterable[T],
    min_chunk_size: int = 100
) -> List[T]:
    """
    Recursively remove duplicates from a (possibly streaming) dataset
    using equality (==) comparison. Single-threaded for minimal overhead.

    Args:
        data_gen (iterable): Stream of objects.
        min_chunk_size (int): Smallest chunk size before recursion stops.

    Returns:
        list: Deduplicated list of unique objects.
    """

    def dedup_sequential(seq: List[T]) -> List[T]:
        """Sequential deduplication using == comparison."""
        unique = []
        for item in seq:
            if not any(item == u for u in unique):
                unique.append(item)
        return unique

    def recursive_dedup(seq: List[T]) -> List[T]:
        """Recursive deduplication logic for large sequences."""
        n = len(seq)
        if n <= min_chunk_size:
            return dedup_sequential(seq)

        mid = n // 2
        left, right = seq[:mid], seq[mid:]

        # Recursively deduplicate left and right halves
        left_unique = recursive_dedup(left)
        right_unique = recursive_dedup(right)

        # Merge results and deduplicate again
        return dedup_sequential(list(chain(left_unique, right_unique)))

    def chunk_generator(gen: Iterable[T], chunk_size: int):
        """Yield successive chunks from a generator."""
        chunk = []
        for item in gen:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    # Process top-level chunks sequentially
    partials = [recursive_dedup(chunk) for chunk in chunk_generator(data_gen, min_chunk_size * 4)]

    # Merge all chunks and deduplicate final results
    combined = chain.from_iterable(partials)
    return dedup_sequential(list(combined))

class Matrix():
    initial_sequence: list[int]
    zero_row: list[int]

    matrix: list[list[int]]
    sum_array: list[int]
    current_sequence: list[int]

    length: int
    iter: int

    
    def set_initial_sequence(self, initial_sequence: list[int]):
        self.initial_sequence = initial_sequence

    def set_zero_row(self, zero_row: list[int]):
        self.zero_row = zero_row

    def __init__(self, sequence: list[int]):
        self.length = len(sequence)
        self.sum_array = [0 for _ in range(self.length)]
        self.set_zero_row(self.sum_array) # Sets zero_row to list of zeros
        self.current_sequence = sequence
        self.matrix = []

    def __update_sum_array(self, sequence: list[int]):
        for column in range(self.iter, self.length): # This can be saved instead of compiled every time
            self.sum_array[column] = self.sum_array[column] + sequence[column]

    def update_matrix_array(self, new_sequence: list[int]):
        s = copy.copy(self.current_sequence)
        for i in range(self.iter+1, self.length):
            s[i] = s[i] - new_sequence[i]
        self.__update_sum_array(s)
        self.matrix.append(s)
        
    @property
    def iter(self):
        return len(self.matrix)
    
    @property
    def is_last(self):
        if self.iter == self.length-1: 
            return True
        return False
    
    def __eq__(self, value: "Matrix") -> bool:
        a = copy.deepcopy(self.matrix)
        b = copy.deepcopy(value.matrix)

        # Squaring the matrixs
        for x in range(self.iter, self.length):
            a.append(self.zero_row)
            b.append(self.zero_row)

        G1 = nx.from_numpy_array(np.array(a))
        G2 = nx.from_numpy_array(np.array(b))
        del a
        del b

        for i in range(len(self.initial_sequence)):
            G1.nodes[i]['weight'] = self.initial_sequence[i]
            G2.nodes[i]['weight'] = self.initial_sequence[i]

        def node_match(n1, n2):
            return n1['weight'] == n2['weight']

        return nx.is_isomorphic(G1, G2, node_match=node_match)
    
    def __str__(self) -> str:
        ret = ""
        for row in range(0, len(self.matrix)):
            ret = ret + f"{self.matrix[row]}"
            if row != len(self.matrix)-1:
                ret = ret + "\n"
        return ret
            


def parse_sequence(s: list[str]) -> list[list[int]]:
    start_matrix = Matrix(copy.copy(s))
    length = len(s)
    start_matrix.set_initial_sequence(copy.copy(s))

    start = [start_matrix]
    out = []

    # This function is run per input matrix and returns every possible row
    def parse_row(matrix: Matrix):
        value = matrix.current_sequence[matrix.iter]
        matrix.current_sequence[matrix.iter] = 0

        combos = distribute_identical_objects(value, matrix.current_sequence)

        def test_graphical(iter: int, sequence: list[int]) -> bool:
            non_zero_indexes = 0
            for x in range(iter+1, length):
                if sequence[x] != 0:
                    non_zero_indexes = non_zero_indexes + 1
            
            if non_zero_indexes == 0: # n=0 is graphical
                return True
            for x in range(iter+1, length):
                if sequence[x] > non_zero_indexes-1:
                    return False
            return True
        
        def parse_combo(combo):
            if test_graphical(matrix.iter, combo):
                m = copy.deepcopy(matrix)
                m.set_initial_sequence(matrix.initial_sequence)
                m.set_zero_row(matrix.zero_row)
                m.update_matrix_array(combo)
                m.current_sequence = combo
                out.append(m)

        futures = []
        with ThreadPoolExecutor(max_workers=MAX_SUB_WORKERS) as executor:
            futures = []
            for row in range(0, len(combos)):
                futures.append(executor.submit(parse_combo, combos[row]))
            for future in as_completed(futures):
                future.result()
        return

    for i in range(0, length-1):
        out: list[Matrix] = []

        futures = []
        with ThreadPoolExecutor(max_workers=MAX_TOP_WORKERS) as executor:
            futures = []
            for matrix in start:
                futures.append(executor.submit(parse_row, matrix))
            for future in as_completed(futures):
                future.result()

        start = []
        start = remove_duplicates_recursive_gen_eq(out)

    for x in start: # adding final zero row
        x.matrix.append(x.zero_row)
    gc.collect()
    return start


import csv

sequences = []
data_lists = ["data/test_data_9.csv"]
#data_lists = ["data/test_data_example.csv"]
for x in data_lists:
    with open(x, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence = str(row["sequence"]).split(" ")
            for i in range(0, len(sequence)):
                sequence[i] = int(sequence[i])
            value = int(row["value"])
            sequences.append([sequence, value])

sequences = tuple(sequences)
total_sequence = len(sequences)

failed = []

broken_adjacency_matrix = []

def per_sequence(idx):
    s = sequences[idx][0]
    value = sequences[idx][1]
    start = time.perf_counter()
    result = parse_sequence(s)
    end = time.perf_counter()
    total_matrices = len(result)
    elapsed_ms = (end - start) * 1000
    print(f"{idx+1}/{total_sequence} n={len(s)} Sequence {s} with target value {value} | Got {total_matrices} | Time: {elapsed_ms:.3f} ms")
    if total_matrices != value:
        print("    Failed test.\n")
        failed.append(f"{s} | {value} | {total_matrices}")
    del result

total_start = time.perf_counter()
futures = []
with ProcessPoolExecutor(max_workers=200) as executor:
    futures = []
    for i in range(0, total_sequence):
        futures.append(executor.submit(per_sequence, i))
    for future in as_completed(futures):
        future.result()
total_end = time.perf_counter()
print(f"Took {total_end-total_start} seconds")

if len(failed) > 0:
    for x in failed:
        print(x)

# print("Solved adjacency Matrix")
# print([0,1,1,1,0])
# print([1,0,1,0,1])
# print([1,1,0,0,0])
# print([1,0,0,0,1])
# print([0,1,0,1,0])


# for unique in range(len(broken_adjacency_matrix)):
#     print("Graph " + str(unique+1))
#     for row in broken_adjacency_matrix[unique]:
#         print(row)
#     print("")
# print("- - - - - - ")

# print(len(broken_adjacency_matrix))

# matrix = []
# x: Matrix
# for x in broken_adjacency_matrix:
#     matrix.append(nx.from_numpy_array(np.array(x.matrix)))
# # # nx.draw(a, with_labels=True, node_color='skyblue', node_size=1000, font_size=12, font_weight='bold')
# # # nx.draw(b, with_labels=True, node_color='skyblue', node_size=1000, font_size=12, font_weight='bold')
# # # nx.draw(c, with_labels=True, node_color='skyblue', node_size=1000, font_size=12, font_weight='bold')
# # # nx.draw(d, with_labels=True, node_color='skyblue', node_size=1000, font_size=12, font_weight='bold')
# # # nx.draw(e, with_labels=True, node_color='skyblue', node_size=1000, font_size=12, font_weight='bold')
# # # plt.title("Graph Visualization from Adjacency Matrix")
# # # plt.show()

# fig, axes = plt.subplots(2, 3, figsize=(10, 5)) # 1 row, 2 columns
# axes = axes.flatten()

# # Draw G1 on the first subplot
# for i, G in enumerate(matrix):
#     ax = axes[i]
#     pos = nx.spring_layout(G) # Choose a layout algorithm
#     nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', node_size=700, font_size=8)
#     ax.set_title(f"Graph {i+1}") # Optional: set a title for each subplot

# plt.tight_layout() # Adjust layout to prevent overlap
# plt.show()

