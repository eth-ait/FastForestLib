from __future__ import division

from abc import ABCMeta, abstractproperty


class TreeNode:
    __metaclass__ = ABCMeta

    def get_split_point(self): pass

    def set_split_point(self, value): pass

    split_point = abstractproperty(get_split_point, set_split_point)

    def get_statistics(self): pass

    def set_statistics(self, value): pass

    statistics = abstractproperty(get_statistics, set_statistics)

    @abstractproperty
    def left_child(self): pass

    @abstractproperty
    def right_child(self): pass


class Tree:

    class _Node(TreeNode):

        _splitPoint = None
        _statistics = None

        _leftChild = None
        _rightChild = None

        def __init__(self, left_child, right_child):
            self._leftChild = left_child
            self._rightChild = right_child

        @property
        def split_point(self):
            return self._splitPoint

        @split_point.setter
        def split_point(self, value):
            self._splitPoint = value

        @property
        def statistics(self):
            return self._statistics

        @statistics.setter
        def statistics(self, value):
            self._statistics = value

        @property
        def left_child(self):
            return self._leftChild

        @property
        def right_child(self):
            return self._rightChild

    def __init__(self, maximum_depth):
        self._root = self._build_tree(maximum_depth)

    def _build_tree(self, maximum_depth, current_depth=0):
        if current_depth >= maximum_depth:
            return None
        left_child = self._build_tree(maximum_depth, current_depth + 1)
        right_child = self._build_tree(maximum_depth, current_depth + 1)
        node = Tree._Node(left_child, right_child)
        return node

    @property
    def root(self):
        return self._root


class ArrayTree:

    class _NodeData:

        def __init__(self):
            self.splitPoint = None
            self.statistics = None

    class _NodeWrapper(TreeNode):

        def __init__(self, tree, index):
            self._tree = tree
            self._index = index

        def _get_node_data(self, index):
            return self._tree._nodes[index]

        @property
        def split_point(self):
            return self._get_node_data(self._index).splitPoint

        @split_point.setter
        def split_point(self, value):
            self._get_node_data(self._index).splitPoint = value

        @property
        def statistics(self):
            return self._get_node_data(self._index).statistics

        @statistics.setter
        def statistics(self, value):
            self._get_node_data(self._index).statistics = value

        @property
        def left_child(self):
            left_index = self._index * 2 + 1
            if left_index >= len(self._tree._nodes):
                return None
            else:
                return ArrayTree._NodeWrapper(self._tree, left_index)

        @property
        def right_child(self):
            right_index = self._index * 2 + 2
            if right_index >= len(self._tree._nodes):
                return None
            else:
                return ArrayTree._NodeWrapper(self._tree, right_index)


    def __init__(self, maximum_depth):
        self._nodes = self._build_tree(maximum_depth)

    def _build_tree(self, maximum_depth, current_depth=0):
        num_of_nodes = 2**maximum_depth - 1
        nodes = []
        for i in xrange(num_of_nodes):
            nodes.append(ArrayTree._NodeData())
        return nodes

    def __len__(self):
        return len(self._nodes)

    @property
    def root(self):
        return ArrayTree._NodeWrapper(self, 0)
