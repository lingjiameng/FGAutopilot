import numpy as np
from DRL.replay_memory.sum_segment_tree import SumSegmentTree
from DRL.replay_memory.min_segment_tree import MinSegmentTree
from DRL.replay_memory.max_segment_tree import MaxSegmentTree

'''
 ' Prioritized experience replay using a SumSegmentTree for prioritized
 ' sampling, and a MinSegmentTree for computing importance sampling (IS)
 ' weights.
 ' Original paper: https://arxiv.org/pdf/1511.05952.pdf
 ' Description (without IS weights):
 '   https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
 ' OpenAI Baselines has an implementation with IS:
 '   https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
 ' Parameters come from the Tuned Parameters table in the PER paper (above).
'''
class PrioritizedReplayMemory():
  '''
   ' Init.
  '''
  def __init__(self, capacity, epsilon = 1e-6, alpha = .6, beta = .4):
    self.epsilon   = epsilon
    self.alpha     = alpha
    self.beta      = beta

    self._sum_tree = SumSegmentTree(capacity)
    self._min_tree = MinSegmentTree(capacity)
    self._max_tree = MaxSegmentTree(capacity)

    # Initial samples (prior to training) get added with this priority.
    self._max_prio = (1.0 + self.epsilon) ** self.alpha

  '''
   ' Number of items in memory.
  '''
  def size(self):
    return self._sum_tree.size()

  '''
   ' Add an item to memory.
  '''
  def add(self, item):
    # The item is added to the sum tree with maximum priority.
    ind = self._sum_tree.add(item, self._max_prio)

    # The priorities are also also kept min and max trees so that the minimum
    # and maximum priorities can be kept track of.
    self._min_tree.add(None, self._max_prio)
    self._max_tree.add(None, self._max_prio)

    return ind

  '''
   ' Get the priority value at an index in the tree.
  '''
  def get_prio(self, ind):
    return self._sum_tree.get_prio(ind)

  '''
   ' Get the maximum priority.
  '''
  def get_max_prio(self):
    return self._max_prio

  '''
   ' Get the minimum priority.
  '''
  def get_min_prio(self):
    return self._min_tree.get_base_prio()

  '''
   ' Get a random memory sample.
   ' Returns an array of samples, each in the following format:
   ' (index, priority, is_weight, transition)
  '''
  def get_random_sample(self, sample_size, priority=True):
    samples_weighted = np.zeros((sample_size, 4), dtype=object)

    # The samples come from the SumSegmentTree.  Each is in the format returned
    # from SegmentTree: (index, priority, transition)
    #
    # The importance samping weight needs to be calculated for each sample, and
    # each IS weight is divided by the maximum IS weight for stability reasons
    # (per the PER paper).
    samples = self._sum_tree.get_random_sample(sample_size, priority)

    # Sum of all priorities.
    prio_sum = self._sum_tree.get_base_prio()

    # Probability of selecting the sample with the minimum priority.
    prob_min = self._min_tree.get_base_prio() / prio_sum

    # Maximum possible IS weight.
    max_is_weight = (prob_min * self._sum_tree.size()) ** -self.beta

    for i in range(sample_size):
      prob_sample = samples[i][1] / prio_sum
      is_weight   = (prob_sample * self._sum_tree.size()) ** -self.beta
      is_weight  /= max_is_weight

      samples_weighted[i] = (samples[i][0], samples[i][1], is_weight, samples[i][2])

    return samples_weighted

  '''
   ' Update the error (and hence, priority) of the item associated with ind.
  '''
  def update(self, ind, error):
    # Convert the error to a priority.  Epsilon is used so that
    # no item has 0 priority.  Alpha is used to determine how much
    # priority is weighted.  For example, an alpha of 0 would mean
    # items are sampled uniformly.
    prio = (error + self.epsilon) ** self.alpha

    # Update the priority in the segment trees.
    self._sum_tree.update(ind, prio)
    self._min_tree.update(ind, prio)
    self._max_tree.update(ind, prio)

    self._max_prio = self._max_tree.get_base_prio()

