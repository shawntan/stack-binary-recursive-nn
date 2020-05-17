import torch
import torch.nn as nn

class RNNOp(nn.Module):
    def __init__(self, nhid, dropout=0.):
        super(RNNOp, self).__init__()
        self.op = nn.Sequential(
            nn.Linear(2 * nhid, nhid),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, left, right):
        return self.op(torch.cat([left, right], dim=-1))

class LSTMOp(nn.Module):
    def __init__(self, nhid, dropout=0.):
        super(LSTMOp, self).__init__()
        assert(nhid % 2 == 0)
        self.hidden_size = nhid // 2
        self.transform = nn.Linear(nhid, 5 * (nhid // 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, left:torch.Tensor, right:torch.Tensor):
        l_h, l_c = left.chunk(2, dim=-1)
        r_h, r_c = right.chunk(2, dim=-1)

        h = torch.cat([l_h, r_h], dim=-1)
        lin_gates, lin_in_c = self.transform(h).split(
            (4 * self.hidden_size, self.hidden_size), dim=-1)
        i, f1, f2, o = torch.sigmoid(lin_gates).chunk(4, dim=-1)
        in_c = torch.tanh(lin_in_c)
        c = i * in_c + f1 * l_c + f2 * r_c
        h = o * torch.tanh(c)
        return torch.cat((h, c), dim=-1)

class Recursive(nn.Module):
    def __init__(self, op,
                 vocabulary_size,
                 hidden_size, padding_idx,
                 parens_id=(0, 1),
                 dropout=0.):
        super(Recursive, self).__init__()
        self.hidden_size = hidden_size
        self.op = op(hidden_size, dropout=dropout)
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.paren_open, self.paren_close = parens_id
        self._recurse = Recursive_(
            self.hidden_size, self.op,
            self.padding_idx, self.embedding,
            self.paren_open, self.paren_close
        )
        self.__dict__['recurse'] = torch.jit.script(self._recurse)

    def forward(self, input):
        return self.recurse(input)

    def __getstate__(self):
        recurse_ = self.recurse
        del self.__dict__['recurse']
        state = self.__dict__.copy()
        self.__dict__['recurse'] = recurse_
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__['recurse'] = torch.jit.script(self._recurse)


class Recursive_(nn.Module):
    def __init__(self, hidden_size, op, padding_idx,
                 embedding, paren_open, paren_close):
        super(Recursive_, self).__init__()
        self.hidden_size = hidden_size
        self.op = op
        self.padding_idx = padding_idx
        self.embedding = embedding
        self.paren_open, self.paren_close = paren_open, paren_close

    def forward(self, input):
        max_length, batch_size  = input.size()
        # Masking business
        length_mask = input != self.padding_idx
        open_mask = input == self.paren_open
        # Extract operations only
        op_mask = length_mask & (~open_mask)
        op_lengths = op_mask.sum(dim=0)

        # 1. Remove all `(` from the sequence.
        op_input = torch.full_like(input[:op_lengths.max()],
                                   self.padding_idx)
        for i in range(batch_size):
            op_input[:op_lengths[i], i] = input[op_mask[:, i], i]
        close_mask = op_input == self.paren_close
        token_mask = (op_input != self.padding_idx) &  (~close_mask)

        # Initialise stack
        stack_height = torch.sum(~close_mask, dim=0).max() + 1
        input_emb = self.embedding(op_input)

        batch_idx = torch.arange(batch_size,
                                 dtype=torch.long, device=input.device)
        stack_ptr = torch.zeros(batch_size,
                                dtype=torch.long, device=input.device)
        stack = torch.zeros(batch_size, stack_height, self.hidden_size,
                            device=input.device)
        for t in range(input_emb.size(0)):
            stack, stack_ptr = self.step(
                batch_idx,
                input_emb[t],
                is_shift=token_mask[t],
                is_reduce=close_mask[t],
                stack=stack, stack_ptr=stack_ptr
            )
        return stack[:, 0]

    def step(self, batch_idx:torch.Tensor, emb_t:torch.Tensor,
             is_shift: torch.Tensor, is_reduce:torch.Tensor,
             stack:torch.Tensor, stack_ptr:torch.Tensor):
        # stack_ptr_ = stack_ptr
        # stack_ptr = stack_ptr_.clone()

        # 2. Batched shift and reduce operations
        # shift
        if is_shift.any():
            shift_stack = stack[is_shift]
            shift_stack_ptr = stack_ptr[is_shift]
            idx = torch.arange(shift_stack.size(0),
                               dtype=shift_stack_ptr.dtype,
                               device=shift_stack_ptr.device)
            shift_stack[idx, shift_stack_ptr] =  emb_t[is_shift]
            stack[is_shift] = shift_stack
            stack_ptr[is_shift] = shift_stack_ptr + 1

        # reduce
        if is_reduce.any():
            reduce_stack = stack[is_reduce]
            reduce_stack_ptr = stack_ptr[is_reduce]
            idx = torch.arange(reduce_stack.size(0),
                               dtype=reduce_stack_ptr.dtype,
                               device=reduce_stack_ptr.device)
            r_child = reduce_stack[idx, reduce_stack_ptr - 1]
            l_child = reduce_stack[idx, reduce_stack_ptr - 2]
            parent = self.op(l_child, r_child)
            reduce_stack[idx, reduce_stack_ptr - 2] = parent
            stack[is_reduce] = reduce_stack
            stack_ptr[is_reduce] = reduce_stack_ptr - 1

        return stack, stack_ptr
if __name__ == "__main__":
    # from recursive import Recursive, RNNOp
    # import torch

    tree = Recursive(RNNOp, 5, 4, padding_idx=4)
    batch_result = tree.forward(torch.tensor([
        [2, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # 2
        [0, 0, 3, 3, 1, 0, 2, 2, 1, 1],  # ( ( 3 3 ) ( 2 2 ) )
        [0, 2, 0, 3, 0, 2, 3, 1, 1, 1]  # ( 2 ( 3 ( 2 3 ) ) )
    ], dtype=torch.long).t())

    # First item (singleton)
    assert (torch.allclose(
        batch_result[0],
        tree.embedding(torch.Tensor([[2]]).long().t())[0]))

    # Second item (balanced tree)
    embs = tree.embedding(torch.Tensor(([3, 3, 2, 2],)).long().t())
    result = tree.op(tree.op(embs[0], embs[1]),
                     tree.op(embs[2], embs[3]))
    assert (torch.allclose(batch_result[1], result))

    # Third item (right linear tree)
    embs = tree.embedding(torch.Tensor([[2, 3, 2, 3]]).long().t())
    result = tree.op(embs[0],
                     tree.op(embs[1],
                             tree.op(embs[2], embs[3])))
    assert (torch.allclose(batch_result[2], result))

    batch_result.sum().backward()