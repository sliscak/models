from torch import sparse, linalg
import torch

def sparse_dot(m1, m2):
    """"
        Sparse dot product
    """
    y = sparse.sum(m1 * m2).to_sparse()
    return y


# def sparse_argmax(m1):
#     argmax = m1.values.argmax()

def sparse_cosine_similarity(a, b, dim=1):
    """"
        Works only for Zero, One and Two dimensional tensors.
    """
    if a.shape == b.shape:
        if len(a.shape) == 1:
            if dim == 0:
                y = sparse_dot(a, b) / (linalg.norm(a) * linalg.norm(b))
                return y
            else:
                raise IndexError("Dimension out of range (expected to be in range of [-1, 0], but got 1)")
        elif len(a.shape) == 2:
            if dim == 0:
                # a = a.to_dense()
                a = torch.unbind(a, 1)

                # b = b.to_dense()
                b = torch.unbind(b, 1)

                y = None
                for i in range(len(a)):
                    sd = (sparse_dot(a[i], b[i]) / (linalg.norm(a[i]) * linalg.norm(b[i]))).unsqueeze(0)
                    if i == 0:
                        y = sd
                    else:
                        y = torch.cat((y, sd), 0)
                # y = sparse_dot(a, b) / (linalg.norm(a) * linalg.norm(b))
                return y
            elif dim == 1:
                # a = a.to_dense()
                a = torch.unbind(a, 0)

                # b = b.to_dense()
                b = torch.unbind(b, 0)

                y = None
                for i in range(len(a)):
                    sd = (sparse_dot(a[i], b[i]) / (linalg.norm(a[i]) * linalg.norm(b[i]))).unsqueeze(0)
                    if i == 0:
                        y = sd
                    else:
                        y = torch.cat((y, sd), 0)
                # y = sparse_dot(a, b) / (linalg.norm(a) * linalg.norm(b))
                return y
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # if dim == 0:
    #     y = sparse_dot(a, b) / (linalg.norm(a) * linalg.norm(b))
    #     # a = a.to_dense()
    #     # b = b.to_dense()
    #     # y = torch.dot(a,b) / (linalg.norm(a) * linalg.norm(b))
    #     y = y.unsqueeze(0)
    # return y


# a = torch.rand(10)
# b = torch.rand(10)
#
# sa = a.to_sparse()
# sb = b.to_sparse()
#
# sim = torch.cosine_similarity(a, b, 0)
# print(sim)
# ssim = sparse_cosine_similarity(sa, sb, 0)
# print(ssim)
# print('')
#
# dot = torch.dot(a, b)
# print(dot)
# sdot = sparse_dot(sa, sb)
# print(sdot)
#
c = torch.rand(2,3, dtype=float)
sc = c.to_sparse()
# sc = sc.to_dense()
print(c.shape)

cs = torch.cosine_similarity(c, c, 1)
print(cs)
print(cs.shape)
#
scs = sparse_cosine_similarity(sc, sc, 1)
print(scs.to_dense())
# # exit()
