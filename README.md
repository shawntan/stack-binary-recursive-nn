# Stack Binary Recursive Neural Networks 
This repository provides an example of Recursive NNs
(as in [Socher et. al., 2011](https://nlp.stanford.edu/pubs/SocherLinNgManning_ICML2011.pdf)),
limited to binary tree structures.
It is implemented in PyTorch, and internally uses a stack to process the
elements of the tree in parallel.
---
### Explanation
Suppose we had some example, input:
```
( ( not a ) ( or d ) )
```

we might turn these into the indices,
```
0 0 2 3 1 0 4 5 1 1
```
where, `(` and `)` are `0` and `1` respectively.

One possible way of processing such a sequence in its tree structure would be
```python
stack = []
for idx in sent:
    if idx == 1:
        # Reduce
        right = stack.pop()
        left = stack.pop()
        stack.append(self.op(left, right))
    else:
        # Shift
        if idx != 0:
            emb = self.embedding.weight[idx]
            stack.append(emb)
```
where `self.op` and  `self.embedding` are modules the recursive operator used to
lookup and compose two child representations into its parent representation.
The naive method would be to process each item in the minibatch as above, and
concatenate them into a minibatch of root representations. 

This method is implemented in `tree_rnn.py`

However, the above approach does not fully utilise the parallelism on GPUs.
Ideally, we should maintain as many stacks as there are instances in a minibatch.

Lets make some observations about the parsing method above:
1. If `idx == 0` or `(`, nothing happens during that iteration. Since we know that the input trees
   are binary then the constituents during a reduce operation is always the top two items on the
   stack.
   
   We can exploit this by first removing all instances of `(` from the minibatch.
   If the trees are deeply nested, this can save many operations, and hopefully 
   bring more shift or reduce operations in parallel.
   
2. Several shift or reduce steps could be happening in parallel for that minibatch.
   
   Using indexing and `.index_put()` we can perform these steps for items in the
   minibatch in parallel.

This method is implemented in `recursive.py`

---

### Example
An example of usage and equivalence in composition order:
```python
from recursive import Recursive, RNNOp
import torch
tree = Recursive(RNNOp, 5, 4, padding_idx=4)
batch_result = tree.forward(torch.tensor([
       [2, 4, 4, 4, 4, 4, 4, 4, 4, 4], # 2 
       [0, 0, 3, 3, 1, 0, 2, 2, 1, 1], # ( ( 3 3 ) ( 2 2 ) )
       [0, 2, 0, 3, 0, 2, 3, 1, 1, 1]  # ( 2 ( 3 ( 2 3 ) ) )
    ], dtype=torch.long).t())

# First item (singleton)
assert(torch.allclose(
    batch_result[0],
    tree.embedding(torch.Tensor([[2]]).long().t())[0]))

# Second item (balanced tree)
embs = tree.embedding(torch.Tensor(([3, 3, 2, 2],)).long().t())
result = tree.op(tree.op(embs[0], embs[1]),
                 tree.op(embs[2], embs[3]))
assert(torch.allclose(batch_result[1], result))

# Third item (right linear tree)
embs = tree.embedding(torch.Tensor([[2, 3, 2, 3 ]]).long().t())
result = tree.op(embs[0],
                 tree.op(embs[1],
                         tree.op(embs[2], embs[3])))
assert(torch.allclose(batch_result[2], result))
```

---
### Logical Inference
Some example usage and training code (`proplog_treernn.py`) is included for the logical inference task introduced in [Bowman et al (2015)](https://arxiv.org/pdf/1506.04834.pdf)
