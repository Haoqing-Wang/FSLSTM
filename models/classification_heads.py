import torch
import torch.nn as nn
from qpth.qp import QPFunction
from torch.autograd import Variable

# one_vs_one
# M = torch.tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
#                   [-1., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
#                   [0., -1., 0., 0., -1., 0., 0., 1., 1., 0.],
#                   [0., 0., -1., 0., 0., -1., 0., -1., 0., 1.],
#                   [0., 0., 0., -1., 0., 0., -1., 0., -1., -1.]]).cuda()  # (K,10)
# L = 10

# one_vs_all
M = torch.tensor([[1., -1., -1., -1., -1.],
                  [-1., 1., -1., -1., -1.],
                  [-1., -1., 1., -1., -1.],
                  [-1., -1., -1., 1., -1.],
                  [-1., -1., -1., -1., 1.]]).cuda()  # (K,5)
L = 5

# ECOC
# M = torch.tensor([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
#                   [-1., -1., -1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
#                   [-1., -1., -1., -1.,  1.,  1.,  1.,  1., -1., -1., -1., -1.,  1.,  1.,  1.],
#                   [-1., -1.,  1.,  1., -1., -1.,  1.,  1., -1., -1.,  1.,  1., -1., -1.,  1.],
#                   [-1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.]]).cuda()  # (K,10)
# L = 15

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda() #(n_batch, m, depth) or (m, depth)
    index = indices.view(indices.size()+torch.Size([1])) #(n_batch, m, 1)
    if len(indices.size()) < 2:
        encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    else:
        encoded_indicies = encoded_indicies.scatter_(2, index, 1)
    return encoded_indicies

def block_diag(m):
    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2).cuda(), d - 3, 1)
    return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))

def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1]*n_dim_to_append))

def computeGramMatrix(A, B): # A*B^T
    """
    Constructs a linear kernel matrix between A and B. We assume that each row in A and B represents a d-dimensional feature vector.
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))
    return torch.bmm(A, B.transpose(1, 2))

def encoding(indices, depth):
    """
    Returns a encoding tensor.
    Parameters:
      indices:  a (m) Tensor.
      depth: a scalar. Represents the number of classes.
    Returns: a (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()  # (m, depth)
    index = indices.view(indices.size()+torch.Size([1]))  # (m, 1)
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)  # (m, depth)
    return torch.mm(encoded_indicies, M)

def ProtoNetHead(query, support, support_labels, n_way, n_shot, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and
    returns the classification score (=L2 distance to each class prototype) on the query set.
    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).

    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    # Compute Prototypes
    labels_train_transposed = support_labels_one_hot.transpose(1,2)  # (tasks_per_batch, n_way, n_support)    support:(tasks_per_batch, n_support, d)
    prototypes = torch.bmm(labels_train_transposed, support)  # (tasks_per_batch, n_way, d)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes))  # (tasks_per_batch, n_way, d)  query:(tasks_per_batch, n_query, d)
    # Distance Matrix Vectorization Trick
    AB = computeGramMatrix(query, prototypes)  # (tasks_per_batch, n_query, n_way)
    AA = (query * query).sum(dim=2, keepdim=True)  # (tasks_per_batch, n_query, 1)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)  # (tasks_per_batch, 1, n_way)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
    logits = -logits
    if normalize:
        logits = logits / d
    return logits

def MetaOptNetHead_Ridge(query, support, support_labels, n_way, n_shot, lambda_reg=50.0, double_precision=False):
    """
    Fits the support set with ridge regression and
    returns the classification score on the query set.
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      lambda_reg: a scalar. Represents the strength of L2 regularization.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

    # Here we solve the dual problem:
    # Note that the classes are indexed by m & samples are indexed by i.
    # min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i

    # where w_m(\alpha) = \sum_i \alpha^m_i x_i,

    # \alpha is an (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support)
    kernel_matrix += lambda_reg * torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()
    block_kernel_matrix = kernel_matrix.repeat(n_way, 1, 1)  # (n_way * tasks_per_batch, n_support, n_support)
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)  # (tasks_per_batch * n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.transpose(0, 1)  # (n_way, tasks_per_batch * n_support)
    support_labels_one_hot = support_labels_one_hot.reshape(n_way * tasks_per_batch, n_support)  # (n_way*tasks_per_batch, n_support)

    G = block_kernel_matrix
    e = -2.0 * support_labels_one_hot

    # This is a fake inequlity constraint as qpth does not support QP without an inequality constraint.
    id_matrix_1 = torch.zeros(tasks_per_batch * n_way, n_support, n_support)
    C = Variable(id_matrix_1)
    h = Variable(torch.zeros((tasks_per_batch * n_way, n_support)))
    dummy = Variable(torch.Tensor()).cuda()  # We want to ignore the equality constraint.

    if double_precision:
        G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]

    else:
        G, e, C, h = [x.float().cuda() for x in [G, e, C, h]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
    # qp_sol = QPFunction(verbose=False)(G, e.detach(), dummy.detach(), dummy.detach(), dummy.detach(), dummy.detach())

    # qp_sol (n_way*tasks_per_batch, n_support)
    qp_sol = qp_sol.reshape(n_way, tasks_per_batch, n_support)
    # qp_sol (n_way, tasks_per_batch, n_support)
    qp_sol = qp_sol.permute(1, 2, 0)
    # qp_sol (tasks_per_batch, n_support, n_way)

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility
    logits = torch.sum(logits, 1)

    return logits

def MetaOptNetHead_SVM(query, support, support_labels, n_way, n_shot, C_reg=0.1, double_precision=False, maxIter=15):
    """
    Fits the support set with multi-class SVM and
    returns the classification score on the query set.

    This is the multi-class SVM presented in:
    On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
    (Crammer and Singer, Journal of Machine Learning Research 2001).
    This model is the classification head that we use for the final version.
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

    # Here we solve the dual problem:
    # Note that the classes are indexed by m & samples are indexed by i.
    # min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
    # s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

    # where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    # and C^m_i = C if m  = y_i,
    # C^m_i = 0 if m != y_i.
    # This borrows the notation of liblinear.

    # \alpha is an (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support)
    id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
    block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)

    block_kernel_matrix += 1.0 * torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support).cuda()
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)  # (tasks_per_batch * n_support, n_support)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)

    G = block_kernel_matrix
    e = -1.0 * support_labels_one_hot
    # print (G.size())
    # This part is for the inequality constraints:
    # \alpha^m_i <= C^m_i \forall m,i
    # where C^m_i = C if m  = y_i,
    # C^m_i = 0 if m != y_i.
    id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
    C = Variable(id_matrix_1)
    h = Variable(C_reg * support_labels_one_hot)
    # print (C.size(), h.size())
    # This part is for the equality constraints:
    # \sum_m \alpha^m_i=0 \forall i
    id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()
    A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
    b = Variable(torch.zeros(tasks_per_batch, n_support))

    if double_precision:
        G, e, C, h, A, b = [x.double().cuda() for x in [G, e, C, h, A, b]]
    else:
        G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]
    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility
    logits = torch.sum(logits, 1)
    return logits

def LS_SVM(query, support, support_labels, n_way, n_shot, C_reg=0.1):
    """
    Fits the support set with multi-class LS-SVM and returns the classification score on the query set.
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    batch, n_query, d = query.size(0), query.size(1), query.size(2)
    n_support = support.size(1)

    support_labels_encode = encoding(support_labels.view(batch * n_support), n_way)  # (batch*n_support, L)
    support_labels_encode = support_labels_encode.view(batch, n_support, L)
    support_labels_encode = support_labels_encode.transpose(1,2)  # (batch, L, n_support)

    inp_support = support.unsqueeze(1).expand(batch, L, n_support, d)
    encode_y = support_labels_encode.unsqueeze(3).expand(batch, L, n_support, d)
    inp_support = (inp_support * encode_y).reshape(batch*L, n_support, d)  # (batch*L, n_support, d)

    G = computeGramMatrix(inp_support, inp_support)+C_reg*torch.eye(n_support).unsqueeze(0).expand(batch*L, n_support, n_support).cuda()
    G = G.reshape(batch, L, n_support, n_support)  # (batch, L, n_support, n_support)
    G = block_diag(G)  # (batch, L*n_support, L*n_support)
    labels_encode = support_labels_encode.unsqueeze(2)  # (batch, L, 1, n_support)
    labels_encode = block_diag(labels_encode)  # (batch, L, L*n_support)
    Zeros = -2.*torch.eye(L).unsqueeze(0).expand(batch, L, L).cuda()
    B = torch.cat([torch.zeros(L).cuda(), torch.ones(L*n_support).cuda()])  # (L*(n_support+1))

    Matrix1 = torch.cat([labels_encode, G], 1)  # (batch, L*(n_support+1), L*n_support)
    Matrix2 = torch.cat([Zeros, labels_encode.transpose(1,2)],1)  # (batch, L*(n_support+1), L)
    Matrix = torch.cat([Matrix2, Matrix1], 2)# (batch, L*(n_support+1), L*(n_support+1))
    B = B.unsqueeze(0).expand(batch, L*(n_support+1)).unsqueeze(2)  # (batch, L*(n_support+1), 1)
    b_inv, _ = torch.solve(B, Matrix)  # (batch, L*(n_support+1), 1)
    solver = b_inv.squeeze(2)  # (batch, L*(n_support+1))

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)
    compatibility = compatibility.float()  # (batch, n_support, n_query)
    bias = solver[:,:L].unsqueeze(2).expand(batch, L, n_query)      # (batch, L, n_query)
    alpha = solver[:,L:].reshape(batch, L, n_support)  # (batch, L, n_support)
    logits = support_labels_encode * alpha  # (batch, L, n_support)
    logits = torch.bmm(logits, compatibility)+bias  # (batch, L, n_query)
    logits = torch.bmm(M.unsqueeze(0).expand(batch, n_way, L), logits)   # (batch, n_way, n_query)  soft
    # logits = torch.bmm(M.unsqueeze(0).expand(batch, n_way, L), torch.sign(logits))  # (batch, n_way, n_query)  no_soft
    logits = logits.transpose(1,2)
    return logits

class ClassificationHead(nn.Module):
    def __init__(self, base_learner='LSSVM', enable_scale=True):
        super(ClassificationHead, self).__init__()
        if ('LSSVM' in base_learner):
            self.head = LS_SVM
        elif ('SVM' in base_learner):
            self.head = MetaOptNetHead_SVM
        elif ('RR' in base_learner):
            self.head = MetaOptNetHead_Ridge
        elif ('NN' in base_learner):
            self.head = ProtoNetHead
        else:
            print("Cannot recognize the base learner type")
            assert(False)
        # Add a learnable scale
        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, query, support, support_labels, n_way, n_shot, **kwargs):
        if self.enable_scale:
            return self.scale * self.head(query, support, support_labels, n_way, n_shot, **kwargs)
        else:
            return self.head(query, support, support_labels, n_way, n_shot, **kwargs)