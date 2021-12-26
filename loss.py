# based on https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch
import torch




def get_gan_losses_fn():
    bce = torch.nn.BCEWithLogitsLoss()

    def real_loss_fn(r_logit):
        return bce(r_logit, torch.ones_like(r_logit))

    def fake_loss_fn(f_logit):
        return bce(f_logit, torch.zeros_like(f_logit))

    return real_loss_fn, fake_loss_fn


def get_hinge_v1_losses_fn():
    def real_loss_fn(r_logit):
        return torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()

    def fake_loss_fn(f_logit):
        return torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()

    return real_loss_fn, fake_loss_fn



def get_lsgan_losses_fn():
    mse = torch.nn.MSELoss()

    def real_loss_fn(r_logit):
        return mse(r_logit, torch.ones_like(r_logit))

    def fake_loss_fn(f_logit):
        return mse(f_logit, torch.zeros_like(f_logit))

    return real_loss_fn, fake_loss_fn


def get_wgan_losses_fn():
    def real_loss_fn(r_logit):
        return -r_logit.mean()

    def fake_loss_fn(f_logit):
        return f_logit.mean()

    return real_loss_fn, fake_loss_fn


def get_adversarial_losses_fn(mode):
    if mode == 'gan':
        return get_gan_losses_fn()
    elif mode == 'hinge_v1':
        return get_hinge_v1_losses_fn()
    elif mode == 'lsgan':
        return get_lsgan_losses_fn()
    elif mode == 'wgan':
        return get_wgan_losses_fn()
    
    
    
# gp ===================================================================================


# ======================================
# =           sample method            =
# ======================================

def _sample_line(real, fake):
    shape = [real.size(0)] + [1] * (real.dim() - 1)
    alpha = torch.rand(shape, device=real.device)
    sample = real + alpha * (fake - real)
    return sample


def _sample_DRAGAN(real, fake):  # fake is useless
    beta = torch.rand_like(real)
    fake = real + 0.5 * real.std() * beta
    sample = _sample_line(real, fake)
    return sample


# ======================================
# =      gradient penalty method       =
# ======================================

def _norm(x):
    norm = x.view(x.size(0), -1).norm(p=2, dim=1)
    return norm


def _one_mean_gp(grad):
    norm = _norm(grad)
    gp = ((norm - 1)**2).mean()
    return gp


def _zero_mean_gp(grad):
    norm = _norm(grad)
    gp = (norm**2).mean()
    return gp


def _lipschitz_penalty(grad):
    norm = _norm(grad)
    gp = (torch.max(torch.zeros_like(norm), norm - 1)**2).mean()
    return gp


def gradient_penalty(f, real, fake, gp_mode, sample_mode):
    sample_fns = {
        'line': _sample_line,
        'real': lambda real, fake: real,
        'fake': lambda real, fake: fake,
        'dragan': _sample_DRAGAN,
    }

    gp_fns = {
        '1-gp': _one_mean_gp,
        '0-gp': _zero_mean_gp,
        'lp': _lipschitz_penalty,
    }

    if gp_mode == 'none':
        gp = torch.tensor(0, dtype=real.dtype, device=real.device)
    else:
        x = sample_fns[sample_mode](real, fake).detach()
        x.requires_grad = True
        pred = f(x)
        grad = torch.autograd.grad(pred, x, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
        gp = gp_fns[gp_mode](grad)
        
    return gp
