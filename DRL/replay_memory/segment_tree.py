from abc import ABC, abstractmethod
import numpy as np
import math

class SegmentTree(ABC):
  '''
   ' Init.
  '''
  def __init__(self, capacity, natural_element):
    # A binary tree is used, where all the items are stored as leaves.  The
    # capacity is required to be a power of 2.
    assert math.log(capacity, 2).is_integer()

    # Maximum number of leaf nodes.
    self._capacity = capacity

    # Total size of the tree (e.g. if there are 8 leaf nodes, the tree has
    # a height of 4 and a size of 2^4-1 = 15).
    self._tree_size = capacity * 2 - 1
    self._memory    = np.zeros(self._capacity, dtype=object)
    self._seg_tree  = np.full(self._tree_size, natural_element)

    # Circular write index.
    self._write = 0

    # Number of leaves currently in the tree.
    self._size = 0

  '''
   ' Get the number of items in memory.
  '''
  def size(self):
    return self._size

  '''
   ' Get the sample at ind (where ind is a seg_tree index).
  '''
  def get_sample_at(self, ind):
    return (ind, self._seg_tree[ind], self._memory[ind - self._capacity + 1])

  '''
   ' Get the priority value at an index in the tree.
  '''
  def get_prio(self, ind):
    #start = self.get_leaf_start_ind()
    #assert ind >= 0 and ind < start + self.size()

    return self._seg_tree[ind]

  '''
   ' Get the priority at the base node.
  '''
  def get_base_prio(self):
    return self.get_prio(0)

  '''
   ' Get the priority of the parent.
  '''
  def get_parent_prio(self, ind):
    return self.get_prio(self.get_parent_index(ind))

  '''
   ' Find the item with prio.
  '''
  @abstractmethod
  def find(self, prio):
    pass

  '''
   ' Add an item to memory and return its tree index.
  '''
  def add(self, item, prio):
    # Circular array -- capacity is never exceeded.
    if self._write == self._capacity:
      self._write = 0

    if self._size < self._capacity:
      self._size += 1

    # Store the item.
    self._memory[self._write] = item

    # Store the priority in the tree, and propagate the sum up to the parents.
    tree_ind = self.get_leaf_start_ind() + self._write
    self.update(tree_ind, prio)

    self._write += 1

    return tree_ind

  '''
   ' Update the priority of the item associated with ind.
  '''
  @abstractmethod
  def update(self, ind, prio):
    pass

  '''
   ' Get the leaf starting point.
  '''
  def get_leaf_start_ind(self):
    # The data are stored in seg_tree, and the priorities are in the leaves
    # (e.g. at the end).  For example, if the capacity is 8, then the 8 leaves
    # are at indices 7 through 14, inclusive.
    return self._capacity - 1

  '''
   ' Given an index, get the parent index.
  '''
  def get_parent_index(self, ind):
    return (ind - 1) // 2

  '''
   ' Given an index, get the left child's index.
  '''
  def get_left_child_index(self, ind):
    return ind * 2 + 1

  '''
   ' Given an index, get the right child's index.
  '''
  def get_right_child_index(self, ind):
    return ind * 2 + 2

