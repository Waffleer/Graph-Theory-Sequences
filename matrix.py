import networkx as nx
import matplotlib.pyplot as plt
import copy
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import numpy as np
import itertools

MAX_WORKERS = 10
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

# Chatgpt
def remove_duplicates_recursive_gen_eq(data_gen: list["Matrix"], max_workers=4, min_chunk_size=100):
    """
    Recursively remove duplicates from a (possibly streaming) dataset
    using equality (==) comparison and parallel processing.

    Args:
        data_gen (iterable or generator): Stream of objects.
        max_workers (int): Number of parallel worker processes.
        min_chunk_size (int): Smallest chunk size before recursion stops.

    Returns:
        list: Deduplicated list of unique objects.
    """

    def dedup_sequential(seq):
        """Sequential deduplication using == comparison."""
        unique = []
        for item in seq:
            if not any(item == u for u in unique):
                unique.append(item)
        return unique

    def recursive_dedup(seq):
        """Recursive deduplication logic."""
        n = len(seq)
        if n <= min_chunk_size:
            return dedup_sequential(seq)

        mid = n // 2
        left, right = seq[:mid], seq[mid:]

        # Parallel recursion for left and right halves
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(recursive_dedup, left)
            right_future = executor.submit(recursive_dedup, right)
            left_unique = left_future.result()
            right_unique = right_future.result()

        # Merge results and deduplicate again
        merged = itertools.chain(left_unique, right_unique)
        return dedup_sequential(list(merged))

    def chunk_generator(gen, chunk_size):
        """Yield successive chunks from a generator."""
        chunk = []
        for item in gen:
            chunk.append(item)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    # Split generator into top-level chunks
    chunks = list(chunk_generator(data_gen, min_chunk_size * 4))

    # Process each chunk recursively (possibly in parallel)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        partials = list(executor.map(recursive_dedup, chunks))

    # Merge and deduplicate final results
    combined = itertools.chain.from_iterable(partials)
    return dedup_sequential(list(combined))


    # with ThreadPoolExecutor(max_workers=3) as executor:
    # # Submit tasks to the thread pool
    
    # futures = [executor.submit(task, i) for i in range(5)]

    # # Retrieve results as they complete
    # for future in futures:
    #     print(future.result())


def parse_sequence_old(s: list[str]) -> list[list[int]]:
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



class Matrix():

    initial_sequence: list[int]
    zero_row: list[int]

    matrix: list[list[int]]
    sum_array: list[int]
    current_sequence: list[int]

    length: int
    iter: int

    @classmethod
    def set_initial_sequence(cls, initial_sequence: list[int]):
        cls.initial_sequence = initial_sequence

    @classmethod
    def set_zero_row(cls, zero_row: list[int]):
        cls.zero_row = zero_row

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

        # first_split: list[list[int]] = []
        # second_split: list[list[int]] = []

        # for l in self.sublist:
        #     first_split.append(self.sum_array[l[0]:l[1]])

        # for l in self.sublist:
        #     second_split.append(value.sum_array[l[0]:l[1]])

        # for split_num in range(0, len(first_split)):
        #     first_split[split_num].sort()
        #     second_split[split_num].sort()
        #     for i in range(0, len(first_split[split_num])):
        #         if first_split[split_num][i] != second_split[split_num][i]:
        #             return False

        # a_cs = copy.copy(self.current_sequence)
        # b_cs = copy.copy(value.current_sequence)
        # a_cs.sort()
        # b_cs.sort()
        # for i in range(self.iter, self.length):
        #     if a_cs[i] != b_cs[i]:
        #         return False


        a = copy.deepcopy(self.matrix)
        b = copy.deepcopy(value.matrix)

        # Squaring the matrixs
        for x in range(self.iter, self.length):
            a.append(self.zero_row)
            b.append(self.zero_row)

        G1 = nx.from_numpy_array(np.array(a))
        G2 = nx.from_numpy_array(np.array(b))

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

    #print(f"Starting Matrix {start_matrix.current_sequence}")

    start = [start_matrix]
    out = []


    def parse_row(matrix: Matrix):
        # Can optimize last iteration since it is deterministic
        # print("")
        # print(f"ITERATION {iter}")
        # print(f"Buckets {sequence}")
        # print(f"Matrix")
        # for x in matrix:
        #     print(x)
        # print("- - - - - - -")

        # Save value ( amount of edges ) of vertex
        value = matrix.current_sequence[matrix.iter]
        matrix.current_sequence[matrix.iter] = 0
        
        # # update_sum_array debug
        # print("???")
        # print(matrix.sum_array)
        # print("+")
        # print(matrix.last_row)
        # print("=")
        # print(matrix.sum_array)
        # print("???")

        # print(f"value {value}")
        # print(f"sequence {sequence}")

        combos = distribute_identical_objects(value, matrix.current_sequence)
        #print(combos)

        #print(combos)


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
                # print("    Is Graphical")
                m = copy.deepcopy(matrix)
                m.update_matrix_array(combo)
                m.current_sequence = combo
                out.append(m)

        futures = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for row in range(0, len(combos)):
                futures.append(executor.submit(parse_combo, combos[row]))
            for future in as_completed(futures):
                future.result()
        executor.shutdown()

        # # Remove non-graphical sequence columns
        # for row_num in range(0, len(combos)):
        #     # print(f"    Working on row {combos[row_num]}")
        #     if test_graphical(matrix.iter, combos[row_num]):
        #         # print("    Is Graphical")
        #         m = copy.deepcopy(matrix)
        #         m.update_matrix_array(combos[row_num])
        #         m.current_sequence = combos[row_num]
        #         out.append(m)
            # else:
            #     print("    Is not Graphical")
        return

        # def per_unique_row(row: list[int]):
        #     m = copy.deepcopy(matrix)
        #     sums_u = copy.deepcopy(sums)
        #     m.append(row_to_matrix_row(iter, row))

        #     if not is_last:
        #         for column in range(iter+1, length): # This can be saved instead of compiled everytime
        #             sums_u[column] = sums_u[column] + row[column]
        #         parse_row(iter+1, row, m, sums_u)
        #     else:
        #         results.append(m)
        #         return

        # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        #     futures = []
        #     for row in unique_rows:
        #         futures.append(executor.submit(per_unique_row, row))
        #     for future in as_completed(futures):
        #         future.result()
        # executor.shutdown()


    for i in range(0, length-1):
        out: list[Matrix] = []
        # print(f"Working on node {i}")
        # print(f"Starting with {len(start)} matrixes")


        futures = []
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for matrix in start:
                futures.append(executor.submit(parse_row, matrix))
            for future in as_completed(futures):
                future.result()
        executor.shutdown()

        # for matrix in start:
        #     parse_row(matrix)

            
                
        # print("All")
        # for m in range(0, len(out)):
        #     print(f"Matrix {m}")
        #     print(out[m])
        #     print(f"sum_array: {out[m].sum_array}")
        #     print(f"Current Sequence: {out[m].current_sequence}")
        # print("")

        # for x in range(0, len(out)):
        #     for y in range(0, len(out)):
        #         if x != y:
        #             print(f"M{x} M{y} {out[x] == out[y]}")

        start = []
        start = remove_duplicates_recursive_gen_eq(out)

        # print(f"{len(start)} Unique Matrixes")
        # for m in range(0, len(start)):
        #     print(f"Matrix {m}")
        #     print(start[m])

        # print("")


    for x in start: # adding final zero row
        x.matrix.append(x.zero_row)
    return start







#s = [5,2,2,2,2,2,5,4,4,4] # 2565
#s = [3,3,2,2,2] # 2
#s = [3,3,2,2,2,2] # 4
#s = [3,3,2,2,2,2,2] # 7
#s = [4, 3, 2, 2, 1, 1, 1] # 7
#s = [2, 2, 2, 2, 2, 1, 1]
#s = [4, 4, 2, 2, 2, 1, 1] # we get 4 website says 5, cant manually find a 5th
#s = [4, 3, 2, 2, 2, 1] # website said 4 we get 3
# s = [3, 2, 2, 2, 2, 1]
# broken_adjacency_matrix = parse_sequence(s)
# for m in broken_adjacency_matrix:
#     print(m)
#     print("")

# print(f"{len(broken_adjacency_matrix)} unique matrixes for s={s}")

import csv

sequences = []
with open("data/test_data_8.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sequence = str(row["sequence"]).split(" ")
        for i in range(0, len(sequence)):
            sequence[i] = int(sequence[i])
        value = int(row["value"])
        sequences.append([sequence, value])

total_sequence = len(sequences)
for sequence in range(0, total_sequence):
    s = sequences[sequence][0]
    value = sequences[sequence][1]
    print(f"{sequence}/{total_sequence} Working on sequence {s} with target value {value} | ", end = "")
    result = parse_sequence(s)
    total_matrices = len(result)
    print(f"Got {total_matrices}")
    if total_matrices != value:
        print("    Failed test.\n")
        break
    




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

