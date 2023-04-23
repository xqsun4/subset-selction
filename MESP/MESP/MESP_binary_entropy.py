"""
    A Unified Perspective on Regularization and Perturbation in Differentiable Subset Selection

    function: select subsets with cardinality k for MESP by regularized relaxation
    
    LML_function: Amos, Brandon, Vladlen Koltun, and J. Zico Kolter. "The limited multi-label projection layer." 
                  arXiv preprint arXiv:1906.08707 (2019).
        
"""
import torch
from torch.autograd import Function, Variable, grad
from torch.nn import Module
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr
import time


from semantic_version import Version
version = Version('.'.join(torch.__version__.split('.')[:3]))
old_torch = version < Version('0.4.0')
starttime = time.time()
torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze()

class LML(Module):
    def __init__(self, N, eps=1e-4, n_iter=100, branch=None, verbose=0):
        super().__init__()
        self.N = N
        self.eps = eps
        self.n_iter = n_iter
        self.branch = branch
        self.verbose = verbose

    def forward(self, x):
        return LML_Function.apply(
            x, self.N, self.eps, self.n_iter, self.branch, self.verbose
        )


class LML_Function(Function):
    @staticmethod
    def forward(ctx, x, N, eps, n_iter, branch, verbose):
        ctx.N = N
        ctx.eps = eps
        ctx.n_iter = n_iter
        ctx.branch = branch
        ctx.verbose = verbose

        branch = ctx.branch
        if branch is None:
            if not x.is_cuda:
                branch = 10
            else:
                branch = 100

        single = x.ndimension() == 1
        orig_x = x
        if single:
            x = x.unsqueeze(0)
        assert x.ndimension() == 2

        n_batch, nx = x.shape
        if nx <= ctx.N:
            y = (1.-1e-5)*torch.ones(n_batch, nx).type_as(x)
            if single:
                y = y.squeeze(0)
            if old_torch:
                ctx.save_for_backward(orig_x)
                ctx.y = y
                ctx.nu = torch.Tensor()
            else:
                ctx.save_for_backward(orig_x, y, torch.Tensor())
            return y

        x_sorted, _ = torch.sort(x, dim=1, descending=True)

        # The sigmoid saturates the interval [-7, 7]
        nu_lower = -x_sorted[:,ctx.N-1] - 7.
        nu_upper = -x_sorted[:,ctx.N] + 7.

        ls = torch.linspace(0,1,branch).type_as(x)

        for i in range(ctx.n_iter):
            r = nu_upper-nu_lower
            I = r > ctx.eps
            n_update = I.sum()
            if n_update == 0:
                break

            Ix = I.unsqueeze(1).expand_as(x) if old_torch else I

            nus = r[I].unsqueeze(1)*ls + nu_lower[I].unsqueeze(1)
            _xs = x[Ix].view(n_update, 1, nx) + nus.unsqueeze(2)
            fs = torch.sigmoid(_xs).sum(dim=2) - ctx.N
            # assert torch.all(fs[:,0] < 0) and torch.all(fs[:,-1] > 0)

            i_lower = ((fs < 0).sum(dim=1) - 1).long()
            J = i_lower < 0
            if J.sum() > 0:
                print('LML Warning: An example has all positive iterates.')
                i_lower[J] = 0

            i_upper = i_lower + 1

            nu_lower[I] = nus.gather(1, i_lower.unsqueeze(1)).squeeze()
            nu_upper[I] = nus.gather(1, i_upper.unsqueeze(1)).squeeze()

            if J.sum() > 0:
                nu_lower[J] -= 7.

        if ctx.verbose >= 0 and np.any(I.cpu().numpy()):
            print('LML Warning: Did not converge.')
            # import ipdb; ipdb.set_trace()

        nu = nu_lower + r/2.
        y = torch.sigmoid(x+nu.unsqueeze(1))
        if single:
            y = y.squeeze(0)

        if old_torch:
            # Storing these in the object may cause memory leaks.
            ctx.save_for_backward(orig_x)
            ctx.y = y
            ctx.nu = nu
        else:
            ctx.save_for_backward(orig_x, y, nu)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        if old_torch:
            x, = ctx.saved_tensors
            y = ctx.y
            nu = ctx.nu
        else:
            x, y, nu = ctx.saved_tensors

        single = x.ndimension() == 1
        if single:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)

        assert x.ndimension() == 2
        assert y.ndimension() == 2
        assert grad_output.ndimension() == 2

        n_batch, nx = x.shape
        if nx <= ctx.N:
            dx = torch.zeros_like(x)
            if single:
                dx = dx.squeeze()
            grads = tuple([dx] + [None]*5)
            return grads

        Hinv = 1./(1./y + 1./(1.-y))
        dnu = bdot(Hinv, grad_output)/Hinv.sum(dim=1)
        dx = -Hinv*(-grad_output+dnu.unsqueeze(1))

        if single:
            dx = dx.squeeze()

        grads = tuple([dx] + [None]*5)
        return grads
    
    
class Identity(Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.theta = Parameter(data = torch.zeros(size=[2000], dtype=torch.float64), requires_grad = True)
        '''change the size'''
        # print('self.theta', self.theta)

  
    def forward(self, k):
        y = LML(N=k)(self.theta)
        return y
    
def MESP_binary(C, k):
    C = torch.tensor(C).to(device)

    model = Identity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    t = 2000 #training steps
    num = k #k-subset selection
    n = 2000
    '''change the n'''
    for i in range(t):
        S = model(k)
        
        S = torch.diag(S).to(device)
        I = torch.eye(n).to(device)
        optimizer.zero_grad()
        tensor1 = torch.matmul(C, S)
        eq = torch.matmul(tensor1, C)
        loss = -torch.log(torch.det(eq +I -S))
        if i%100==0:
            print('-'*10)
            print('Epoch:', i, '; Loss:', loss.data.cpu().numpy())
        
        loss.backward()
        optimizer.step()
        # if i==0:
        #     print('loss', loss)
        #     print('S', S)
        #     print('gradient', model.theta.grad)
            
        # if i%500==0:
        #     print('-'*10, 'iteration time', i, '-'*10)
        #     theta = model.theta.cpu().detach().numpy()
        #     print(theta)

        
    # print('S:', S)
    torch.cuda.empty_cache()
    theta_e = model.theta.cpu().detach().numpy()
    
    return theta_e


def PSD_generating(k):
    matrixSize = k
    A = np.random.rand(matrixSize, matrixSize)
    B = np.dot(A, A.transpose())
    return B

if __name__ == "__main__":

    import scipy.io as scio
    import heapq
    
    #data dimension = 90, 124, 2000
    datafile = 'D://xiangqian_exp2//MESP//data90' 
    n = 2000
    '''change the num'''
    data = scio.loadmat(datafile)
    C = np.array(data['C'])
    gamma = 0.05
    trun_C = C*gamma 
    
    k = 120
    theta = MESP_binary(trun_C, k)
    # S = k*scipy.special.softmax(theta)
    S = heapq.nlargest(k, range(len(theta)), theta.__getitem__)
    CS = np.log(np.linalg.det(C[np.ix_(S, S)])) # log determinant
    # print(theta)
    print('-'*10, 'binary entropy method', '-'*10)
    print('gamma is ', gamma)
    print('C dimension is:', n, '; subset dimension is:', k)
    print('approximate subset:', S)
    theta = torch.tensor(theta)
    y = LML(N=k)(theta).data.numpy()
    print('appro subset prob', y[S])
    # print(heapq.nlargest(k, theta))
    print('log maxmimum entropy:', CS)
    

endtime = time.time()
print ('time:', endtime - starttime) 