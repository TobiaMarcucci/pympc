from __future__ import absolute_import, division, print_function


class Polyhedron(object):
    __slots__ = ["A", "b"]
    def __init__(self, A, b):
        self.A = np.asarray(A)
        self.b = np.asarray(b).reshape(-1)
        assert self.A.shape[0] == self.b.size

    def contains(self, point):
        return np.all(self.A.dot(np.asarray(point)) <= self.b)


class Element(object):
    __slots__ = ["polyhedron", "data"]
    def __init__(self, polyhedron, data=None):
        self.polyhedron = polyhedron
        self.data = data

    def applies_to(self, point):
        return self.polyhedron.contains(point)


class NDPiecewise(object):
    def __init__(self, elements):
        self.elements = elements

    def lookup(self, point):
        for element in self.elements:
            if element.applies_to(point):
                return element
        return None

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)
