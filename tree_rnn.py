import torch
import torch.nn as nn
# import torch.nn.functional as F


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
        self.transform = nn.Linear(2 * nhid, 5 * nhid)

    def forward(self, left, right):

        if isinstance(left, tuple):
            h_left, c_left = left
        else:
            h_left, c_left = left, torch.zeros_like(left)

        if isinstance(right, tuple):
            h_right, c_right = right
        else:
            h_right, c_right = right, torch.zeros_like(right)

        h = torch.cat([h_left, h_right], dim=-1)
        i_, f1_, f2_, o_, u_ = self.transform(h).chunk(5, dim=-1)
        i = torch.sigmoid(i_)
        f1 = torch.sigmoid(f1_)
        f2 = torch.sigmoid(f2_)
        o = torch.sigmoid(o_)
        u = torch.tanh(u_)
        c = i * u + f1 * c_left + f2 * c_right
        h = o * torch.tanh(c)
        return (h, c)


class TreeRNN(nn.Module):
    def __init__(self, ntoken, nhid, parens_id=(0, 1), dropout=0.0, op=RNNOp):
        super(TreeRNN, self).__init__()
        self.op = op(nhid)
        self.padding_idx = ntoken - 1
        self.embedding = nn.Embedding(ntoken, nhid,
                                      padding_idx=self.padding_idx)
        self.paren_open, self.paren_close = parens_id
        # self.embedding = nn.Embedding(ntoken, nhid)

    def forward(self, input):
        lens = (input != self.padding_idx).sum(0)
        parsed_batch = torch.stack([
            self.parse(input[:, i], lens[i])
            for i in range(lens.size(0))
        ])
        return parsed_batch

    def parse(self, sent, length):
        stack = []
        # disp_stack = []
        for idx in sent[:length]:
            if idx == self.paren_close:
                right = stack.pop()
                left = stack.pop()
                stack.append(self.op(left, right))
                # r = disp_stack.pop()
                # l = disp_stack.pop()
                # disp_stack.append((l, r))
            else:
                if idx != self.paren_open:
                    emb = self.embedding.weight[idx]
                    stack.append(emb)
                    # disp_stack.append(idx.item())

        if isinstance(stack[0], tuple):
            return stack[0][0]
        else:
            return stack[0]


if __name__ == "__main__":
    tree = TreeRNN(5, 50)
    print(tree(torch.Tensor([[0, 0, 2, 3, 1, 0, 2, 2, 1, 1],
                             [2, 4, 4, 4, 4, 4, 4, 4, 4, 4]]).long().t()))
    print(tree(torch.Tensor([[2]]).long().t()))



