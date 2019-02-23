import numpy as np
from DRL.replay_memory.segment_tree import SegmentTree

'''
 ' Data structure for retrieving values proportionately based on their priorities.
'''
class SumSegmentTree(SegmentTree):
  '''
   ' Init.
  '''
  def __init__(self, capacity):
    super().__init__(capacity, 0.0)

  '''
   ' Select an item proportionally based on sum.
  '''
  def find(self, sum):
    ind = self._find(0, sum)

    return self.get_sample_at(ind)

  '''
   ' Recursive retrieve based on sum.
  '''
  def _find(self, ind, sum):
    if ind >= self.get_leaf_start_ind():
      return ind

    lInd = self.get_left_child_index(ind)
    rInd = self.get_right_child_index(ind)

    if sum <= self._seg_tree[lInd]:
      return self._find(lInd, sum)
    return self._find(rInd, sum - self._seg_tree[lInd])

  '''
   ' Get a random sample of memory.
  '''
  def get_random_sample(self, sample_size, priority=True):
    sample = np.zeros((sample_size, 3), dtype=object)

    if priority:
      # Sample using priorities.
      for i in range(sample_size):
        sample[i] = self.find(np.random.uniform(0, self.get_base_prio()))
    else:
      # Sample in a purely random manner.
      indices = np.random.choice(self.size(), sample_size)
      start   = self.get_leaf_start_ind()

      for i in range(sample_size):
        sample[i] = self.get_sample_at(start + indices[i])

    return sample

  '''
   ' Update the priority of the item associated with ind.
  '''
  def update(self, ind, prio):
    #start = self.get_leaf_start_ind()
    #assert ind >= start and ind < start + self.size()

    # The change in priority needs to be propagated up the tree.
    old_prio = self.get_prio(ind)
    delta    = prio - old_prio

    self._seg_tree[ind] = prio
    self._propagate_prio(ind, delta)

  '''
   ' Update the priority (sums) starting at ind and moving up the tree.
  '''
  def _propagate_prio(self, ind, delta):
    pInd = self.get_parent_index(ind)

    self._seg_tree[pInd] += delta

    if pInd != 0:
      self._propagate_prio(pInd, delta)

