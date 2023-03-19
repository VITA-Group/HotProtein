import torch

def tensor_clamp(t, min, max, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min
    res.data[idx] = min[idx]
    idx = res.data > max
    res.data[idx] = max[idx]

    return res

def linfball_proj(center, radius, t, in_place=True):
    return tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)

def PGD_classification(x, model=None, labels=None, steps=1, gamma=0.1, eps=(1/255), randinit=False, clip=False, num_classes=None):
    
    # Compute loss
    x_adv = x.clone().detach()
    x_adv.requires_grad = True
    if randinit:
        # adv noise (-eps, eps)
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = x_adv.cuda()
    x = x.cuda()

    for t in range(steps):

        out = model(x_adv)
        loss_adv0 = torch.nn.functional.cross_entropy(out.view(out.shape[0], num_classes), labels)
        # loss_adv0 = torch.mean(out)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))

        if clip:
            linfball_proj(x, eps, x_adv, in_place=True)

    return x_adv


def PGD_classification_amino(x, model=None, labels=None, steps=1, gamma=0.1, eps=(1/255), randinit=False, clip=False, num_classes=None, top=10):
    
    # Compute loss
    x_adv = x.clone().detach()
    x_adv.requires_grad = True
    if randinit:
        # adv noise (-eps, eps)
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = x_adv.cuda()
    x = x.cuda()

    for t in range(steps):

        out = model(x_adv)
        loss_adv0 = torch.nn.functional.cross_entropy(out.view(out.shape[0], num_classes), labels)
        # loss_adv0 = torch.mean(out)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        update = gamma * torch.sign(grad0.data)
        index = torch.randperm(x_adv.shape[0], device=x_adv.device)[:top]
        x_adv.data.index_add_(0, index, update[index])

        if clip:
            linfball_proj(x, eps, x_adv, in_place=True)

    return x_adv


def PGD_regression(x, model=None, labels=None, steps=1, gamma=0.1, eps=(1/255), randinit=False, clip=False, num_classes=None):
    
    # Compute loss
    x_adv = x.clone().detach()
    x_adv.requires_grad = True
    if randinit:
        # adv noise (-eps, eps)
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = x_adv.cuda()
    x = x.cuda()

    for t in range(steps):

        out = model(x_adv)
        loss_adv0 = torch.nn.functional.mse_loss(out.view(out.shape[0], 1), labels)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))

        if clip:
            linfball_proj(x, eps, x_adv, in_place=True)

    return x_adv


def PGD_regression_amino(x, model=None, labels=None, steps=1, gamma=0.1, eps=(1/255), randinit=False, clip=False, num_classes=None, top=10):
    
    # Compute loss
    x_adv = x.clone().detach()
    x_adv.requires_grad = True
    if randinit:
        # adv noise (-eps, eps)
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = x_adv.cuda()
    x = x.cuda()

    for t in range(steps):

        out = model(x_adv)
        loss_adv0 = torch.nn.functional.mse_loss(out.view(out.shape[0], 1), labels)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        update = gamma * torch.sign(grad0.data)
        index = torch.randperm(x_adv.shape[0], device=x_adv.device)[:top]
        x_adv.data.index_add_(0, index, update[index])

        if clip:
            linfball_proj(x, eps, x_adv, in_place=True)

    return x_adv

