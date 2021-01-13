from torch import sparse, linalg

def sparse_dot(m1, m2):
    """"
        Sparse dot product
    """
    y = sparse.sum(m1 * m2)
    return y

# def sparse_argmax(m1):
#     argmax = m1.values.argmax()

def sparse_cosine_similarity(a, b):
    y = sparse_dot(a,b) / (linalg.norm(a) * linalg.norm(b))
    return y

import torch
a = torch.rand(10)
b = torch.rand(10)

sa = a.to_sparse()
sb = b.to_sparse()

sim = torch.cosine_similarity(a, b, 0)
print(sim)
ssim = sparse_cosine_similarity(sa, sb)
print(ssim)

dot = torch.dot(a, b)
print(dot)
sdot = sparse_dot(sa, sb)
print(sdot)
