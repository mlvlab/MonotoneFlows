'''
Adapted from Implicit Normalizing Flow (ICLR 2021).
Link: https://github.com/thu-ml/implicit-normalizing-flows/blob/master/lib/layers/broyden.py
'''


import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np 
import math
import pickle
import sys
import os
from scipy.optimize import root
import time
from termcolor import colored

import logging

logger = logging.getLogger()



def find_fixed_point_noaccel(f, x0, threshold=1000, eps=1e-5):
    b = x0.size(0)
    b_shape = (b,)
    for _ in range(1, len(x0.shape)):
        b_shape = b_shape + (1,)
    alpha = 0.5 * torch.ones(b_shape, device=x0.device)
    x, x_prev = (1-alpha)*x0 + (alpha)*f(x0), x0
    i = 0
    tol = eps + eps * x0.abs()

    best_err = 1e9 * torch.ones(b_shape, device=x0.device)
    best_iter = torch.zeros(b_shape, dtype=torch.int64, device=x0.device)
    while True:
        fx = f(x)
        err_values = torch.abs(fx - x) / tol
        cur_err = torch.max(err_values.view(b, -1), dim=1)[0].view(b_shape)

        if torch.all(cur_err < 1.):
            break
        alpha = torch.where(torch.logical_and(cur_err >= best_err, i >= best_iter + 30),
            alpha * 0.9,
            alpha)
        alpha = torch.max(alpha, 0.1*torch.ones_like(alpha))
        best_iter = torch.where(torch.logical_or(cur_err < best_err, i >= best_iter + 30),
            i * torch.ones(b_shape, dtype=torch.int64, device=x0.device),
            best_iter)
        best_err = torch.min(best_err, cur_err)

        x, x_prev = (1-alpha)*x + (alpha)*fx, x
        i += 1
        if i > threshold:
            dx = torch.abs(f(x) - x)
            rel_err = torch.max(dx/tol).item()
            abs_err = torch.max(dx).item()
            if rel_err > 3 or abs_err > 3 * max(eps, 1e-9):
                logger.info('Relative/Absolute error maximum: %.10f/%.10f' % (rel_err, abs_err))
                logger.info('Iterations exceeded %d for fixed point noaccel.' % (threshold))
            break
    return x


def find_fixed_point(f, x0, threshold=1000, eps=1e-5):
    b = x0.size(0)
    def g(w):
        return f(w.view(x0.shape)).view(b, -1)

    with torch.no_grad():
        X0 = x0.view(b, -1)
        X1 = g(X0)
        Gnm1 = X1
        dXnm1 = X1 - X0
        Xn = X1

        tol = eps + eps * X0.abs()
        best_err = math.inf
        best_iter = 0
        n = 1
        while n < threshold:
            Gn = g(Xn)
            dXn = Gn - Xn
            cur_err = torch.max(torch.abs(dXn) / tol).item()
            if cur_err <= 1.:
                break
            if cur_err < best_err:
                best_err = cur_err
                best_iter = n
            elif n >= best_iter + 10:
                break

            d2Xn = dXn - dXnm1
            d2Xn_norm = torch.linalg.vector_norm(d2Xn, dim=1)
            mult = (d2Xn * dXn).sum(dim=1) / (d2Xn_norm**2 + 1e-8)
            mult = mult.view(b, 1)
            Xnp1 = Gn - mult*(Gn - Gnm1)

            dXnm1 = dXn
            Gnm1 = Gn
            Xn = Xnp1
            n = n + 1

        rel_err = torch.max(torch.abs(dXn)/tol).item()
        if rel_err > 1:          
            abs_err = torch.max(torch.abs(dXn)).item()
            if rel_err > 10:
                return find_fixed_point_noaccel(f, x0, threshold=threshold, eps=eps)
            else:
                return find_fixed_point_noaccel(f, Xn.view(x0.shape), threshold=threshold, eps=eps)
        else:
            return Xn.view(x0.shape)


class RootFind(torch.autograd.Function):
    @staticmethod
    def banach_find_root(Gnet, x, *args):
        eps = args[-2]
        threshold = args[-1]    # Can also set this to be different, based on training/inference
        g = lambda y: x - Gnet(y)
        y_est = find_fixed_point(g, x, threshold=threshold, eps=eps)
        return y_est.clone().detach()

    @staticmethod
    def forward(ctx, Gnet, x, method, *args):
        ctx.args_len = len(args)
        with torch.no_grad():
            y_est = RootFind.banach_find_root(Gnet, x, *args)

            # If one would like to analyze the convergence process (e.g., failures, stability), should
            # insert here or in broyden_find_root.
            return y_est

    @staticmethod
    def backward(ctx, grad_y):
        assert 0, 'Cannot backward to this function.'
        grad_args = [None for _ in range(ctx.args_len)]
        return (None, None, grad_y, None, *grad_args)








'''
Provides backward propagation for the implicit mapping F^-1(x).
'''
class MonotoneBlockBackward(torch.autograd.Function):
    """
    A 'dummy' function that does nothing in the forward pass and perform implicit differentiation
    in the backward pass. Essentially a wrapper that provides backprop for the `MonotoneBlock` class.
    You should use this class in MonotoneBlock's forward() function by calling:

        MonotoneBlockBackward.apply(self.func, ...)

    """
    @staticmethod
    def forward(ctx, Gnet, y, x, *args):
        ctx.save_for_backward(y, x)
        ctx.Gnet = Gnet
        ctx.args = args
        return y

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        y, x = ctx.saved_tensors
        args = ctx.args
        method, eps, threshold = args[-3:]

        Gnet = ctx.Gnet
        y = y.clone().detach().requires_grad_()

        with torch.enable_grad():
            Gy = Gnet(y)

        def h(x_):
            y.grad = None
            Gy.backward(x_, retain_graph=True)
            xJ = y.grad.clone().detach()
            y.grad = None
            return xJ

        dl_dh = RootFind.apply(h, grad, method, eps, threshold)
        dl_dx = dl_dh

        grad_args = [None for _ in range(len(args))]
        return (None, dl_dh, dl_dx, *grad_args)
