class Node:

    feature = None
    threshold = None

    _leftChild = None
    _rightChild = None

    def __init__(self, left_child, right_child):
        self._leftChild = left_child
        self._rightChild = right_child

    @property
    def left_child(self):
        return self._leftChild

    @property
    def right_child(self):
        return self._rightChild
