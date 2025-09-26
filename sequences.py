



class Node:
    def __init__(self, name: str = "", edges: list = []):
        self.edges = edges
        self.name = name

    @property
    def degree(self):
        return len(self.edges)
    
    def __str__(self):
        return self.name



a = Node([Node(name = "b")])


print(a.degree)