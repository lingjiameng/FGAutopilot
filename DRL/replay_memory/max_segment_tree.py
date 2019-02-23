from DRL.replay_memory.segment_tree import SegmentTree

'''
 ' A SegmentTree where each node is the maximum of its children.
'''
class MaxSegmentTree(SegmentTree):
  '''
   ' Init.
  '''
  def __init__(self, capacity):
    super().__init__(capacity, float('-inf'))

  '''
   ' Select an item based on priority.
  '''
  def find(self, prio):
    ind = self._find(0, prio)

    return self.get_sample_at(ind)

  '''
   ' Recursive retrieve based on priority.
  '''
  def _find(self, ind, prio):
    if ind >= self.get_leaf_start_ind():
      return ind

    l_ind = self.get_left_child_index(ind)

    if prio <= self._seg_tree[l_ind]:
      return self._find(l_ind, prio)

    return self._find(self.get_right_child_index(ind), prio)

  '''
   ' Update the priority of the item associated with ind, and propagate the
   ' priority up the tree.
  '''
  def update(self, ind, prio):
    #start = self.get_leaf_start_ind()
    #assert ind >= start and ind < start + self.size()

    self._seg_tree[ind] = prio
    self._propagate_prio(ind)

  '''
   ' Update the priority (max) starting at ind and moving up the tree.
  '''
  def _propagate_prio(self, ind):
    # Parent.
    p_ind  = self.get_parent_index(ind)
    p_prio = self._seg_tree[p_ind]

    # Parent's left child.
    l_ind  = self.get_left_child_index(p_ind)
    l_prio = self._seg_tree[l_ind]

    # Parent's right child.
    r_ind  = self.get_right_child_index(p_ind)
    r_prio = self._seg_tree[r_ind]

    # Maximum child priority.
    child_prio = max(l_prio, r_prio)

    # If the parent's prio is already the maximum of its children's, then the
    # propagation need go no further.
    if child_prio != p_prio:
      self._seg_tree[p_ind] = child_prio

      if p_ind != 0:
        self._propagate_prio(p_ind)

