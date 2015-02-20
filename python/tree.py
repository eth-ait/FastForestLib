from node import Node


class Tree:

    _root = None

    def __init__(self, maximum_depth):
        self._root = self._build_tree(maximum_depth)

    def _build_tree(self, maximum_depth, current_depth=0):
        if current_depth >= maximum_depth:
            return None
        left_child = self._build_tree(maximum_depth, current_depth + 1)
        right_child = self._build_tree(maximum_depth, current_depth + 1)
        node = Node(left_child, right_child)
        return node

    @property
    def root(self):
        return self._root
