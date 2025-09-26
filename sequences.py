import copy



class GraphReductionFailed(Exception):
    def __init__(self, message):            
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        # Now for your custom code...


# class Node:
#     edges: list = []

#     def __init__(self, name: str, edgelist: list):
#         self.name = name
#         self.edges = edgelist
#         print(name)
#         print(edgelist)
#         for node in edgelist:
#             print("node: " + node.name)
#             node.edges.append(self)

#     @property
#     def degree(self):
#         return len(self.edges)
    
#     def __str__(self):
#         return f"{self.name}"
    
class Sequence:


    def __init__(self, sequence: list[int], name: str = ""):
        self.name = name
        sequence.sort(reverse=True)
        self.__sequence = sequence

    @property
    def sequence(self) -> list[int]:
        return copy.deepcopy(self.__sequence)
    
    @property
    def order(self):
        return len(self.sequence)

    @property
    def size(self):
        return (self.__arraySum(self.sequence)/2)
    
    @property
    def sum(self) -> int:
        return self.__arraySum(self.sequence)
    
    @property
    def is_multigraph(self) -> bool:
        if ((self.sum % 2) != 0):
            print("Sum of degrees is not divisible by 2.")
            return False
        return True
    
    @property
    def is_graph(self) -> bool:
        if (not self.is_multigraph):
            return False
        
        try:
            self.__reduce_graph(self.sequence, 0)
        except GraphReductionFailed as e:
            print(e)
            return False
        return True

    def __arraySum(self, arr: list[int]) -> int:
        s = 0
        for x in arr:
            s = s + x
        return s
    

    
    def __formatSubset(self, arr: list[int]) -> str:
        ret = ""
        arr = str(arr)
        diff = len(str(self.sequence))-len(arr)
        
        for i in range(0, diff):
            ret = ret + " "
        return ret + arr

    def __reduce_graph(self, arr: list[int], iter: int) -> list[int]:
        iter+=1
        arr.sort(reverse=True)
        print(f"iteration: {iter} {self.__formatSubset(arr)}")
        if (len(arr) == 0):
            return []
        
        if (self.__arraySum(arr) == 0):
            return []
        
        if (len(arr) == 1 and arr[0] != 0):
            raise GraphReductionFailed(f"Sequence with one vertex has more than one edge. {arr}")
        first = arr[0]
        ret = arr
        ret.pop(0)

        # Reduce graph
        for i in range(0, first):
            try:
                reduction = ret[i]-1
            except IndexError:
                raise GraphReductionFailed(f"Not enough nodes to match highest Degree. {first} {arr}")
            if (reduction < 0):
                raise GraphReductionFailed(f"Degree became less than zero. {first} {arr}")
            ret[i] = reduction

        try:
            self.__reduce_graph(ret, iter)
        except GraphReductionFailed:
            raise # Passes error up the stack
    
    def __str__(self):
        return f"{self.name} = {self.sequence}"
        

a = [10,10,7,7,7,7,4,5,2,1,2,2,5,1] # Is a graph
b = [6,4,3,2,2,2,1] # Is a graph

s = Sequence(b, name="s")
print(s.is_graph)

