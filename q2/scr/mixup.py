import torch
import numpy as np

def mixup_data(x, y, alpha=1.0,
    # use_cuda=True, device="cpu"
    ):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
        # lam = min(lam, 1-lam)
    else:
        lam = 1.
    batch_size = x.size()[0]

    # if use_cuda:
    #     index = torch.randperm(batch_size).to(device)
    # else:
    #     index = torch.randperm(batch_size)
    index = torch.randperm(batch_size).to(x.device)

    ## SYM
    # mixed_x = lam * x + (1 - lam) * x[index,:]
    # mixed_y = (1 - lam) * x + lam * x[index,:]
    # mixed_image  = torch.cat([mixed_x,mixed_y], 0)
    # y_a, y_b = y, y[index]
    # mixed_label  = torch.cat([y_a,y_b], 0)


    ## Reduce batch size
    # new_batch_size = batch_size // 2
    # x_i = x[ : new_batch_size]
    # x_j = x[new_batch_size : ]
    # y_a = y[ : new_batch_size]
    # y_b = y[new_batch_size : ]
    # mixed_x = lam * x_i + (1 - lam) * x_j


    ## NO SYM
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]

    ## Only Alpha
    # mixed_x = 0.5 * x + (1 - 0.5) * x[index,:]
    # mixed_image  = mixed_x
    # y_a, y_b = y, y[index]
    # ind_label = torch.randint_like(y, 0,2)
    # mixed_label  = ind_label * y_a + (1-ind_label) * y_b

    ## Reduce batch size and SYM
    # new_batch_size = batch_size // 2
    # x_i = x[ : new_batch_size]
    # x_j = x[new_batch_size : ]
    # y_a = y[ : new_batch_size]
    # y_b = y[new_batch_size : ]
    # mixed_x = lam * x_i + (1 - lam) * x_j
    # mixed_y = (1 - lam) * x_i + lam * x_j
    # mixed_x  = torch.cat([mixed_x,mixed_y], 0)
    # y_b = torch.cat([y_b,y_a], 0)
    # y_a = y


    # return mixed_image, mixed_label, lam
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    # sigmoid = 1.0/(1 + math.exp( 5 - 10*lam))
    # sigmoid = 4.67840515/(5.85074311 + math.exp(6.9-10.2120858*lam))
    # sigmoid = 1.531 /(1.71822 + math.exp(6.9-12.2836*lam))
    # return lambda criterion, pred: sigmoid * criterion(pred, y_a) + (1 - sigmoid) * criterion(pred, y_b)

    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)