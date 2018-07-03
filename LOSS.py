from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)



class MyLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(MyLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce
    def make_mask(self,input,targ,flag):
        targ=targ.data.numpy().tolist()
        #targ=targ.unsqueeze(-1)
        #print(targ)
        (x1,x2)=input.size()
        mask=torch.FloatTensor(x1,x2).zero_()
        for i in range(len(targ)):
            #print(targ[i],flag)
            if targ[i]==flag:
                mask[i][targ[i]]=1.0
            else:
                mask[i][targ[i]]=1.0
        return mask
    def forward(self, input, target,flag):
        _assert_no_grad(target)
        
        mask=Variable(self.make_mask(input, target,flag),requires_grad=True)
        loss=(torch.log(input)*mask).sum()
        return -loss

if __name__=='__main__':
    m = nn.Softmax()
    loss = MyLoss()
    input = Variable(torch.randn(4, 5), requires_grad=True)
    target = Variable(torch.LongTensor([1, 0, 4, 4]))
    output = loss(m(input), target,0)
    print(output)
    output.backward()
