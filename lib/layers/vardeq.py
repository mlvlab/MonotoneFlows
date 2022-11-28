import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lib.layers as layers
import lib.layers.iresblock as iresblock
import lib.layers.solvers as solvers



def safe_log(x):
    return torch.log(x.clamp(min=1e-22))


def clone_primitive(m):
    with torch.no_grad():
        if callable(getattr(m, "build_clone", None)):
            return m.build_clone()
        if isinstance(m, nn.Conv2d):
            c = nn.Conv2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding,
                bias=m.bias is not None, device=m.weight.device, dtype=m.weight.dtype)
            c.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                c.bias.data.copy_(m.bias.detach())
            return c
        if isinstance(m, nn.GroupNorm):
            c = nn.GroupNorm(m.num_groups, m.num_channels, m.eps, m.weight is not None, device=m.weight.device, dtype=m.weight.dtype)
            if m.weight is not None:
                c.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                c.bias.data.copy_(m.bias.detach())
            return c
        if isinstance(m, nn.ReLU):
            return nn.ReLU()
        if isinstance(m, nn.Identity):
            return nn.Identity()
        raise NotImplementedError()


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, idim, *args):
        super(Bottleneck, self).__init__()

        if in_channels is None:
            self.nnet = args[0]
            self.match_channel = args[1]
        else:
            assert in_channels % 16 == 0
            assert idim % 16 == 0

            nnet_list = []
            nnet_list.append(nn.GroupNorm(in_channels // 16, in_channels))
            nnet_list.append(nn.ReLU())
            nnet_list.append(nn.Conv2d(in_channels, idim, (3,3), stride=1, padding=1))
            nnet_list.append(nn.GroupNorm(idim // 16, idim))
            nnet_list.append(nn.ReLU())
            nnet_list.append(nn.Conv2d(idim, idim, (1,1)))
            nnet_list.append(nn.GroupNorm(idim // 16, idim))
            nnet_list.append(nn.ReLU())
            nnet_list.append(nn.Conv2d(idim, out_channels, (3,3), stride=1, padding=1))
            self.nnet = nn.Sequential(*nnet_list)

            if in_channels != out_channels:
                self.match_channel = nn.Conv2d(in_channels, out_channels, (3,3), stride=1, padding=1)
            else:
                self.match_channel = nn.Identity()

    def forward(self, x):
        y = self.match_channel(x)
        z = self.nnet(x)
        return y + z

    def build_clone(self):
        nnet_list = []
        for nnet in self.nnet:
            nnet_list.append(clone_primitive(nnet))
        nnet = nn.Sequential(*nnet_list)
        match_channel = clone_primitive(self.match_channel)
        return Bottleneck(None, None, None, nnet, match_channel)


class VarDeqConditioningLayer(nn.Module):
    def __init__(self, in_channels, ydim=64, idim=512, nblocks=3, *args):
        super(VarDeqConditioningLayer, self).__init__()

        if in_channels is None:
            self.nnet = args[0]
        else:
            assert ydim % 16 == 0
            assert idim % 16 == 0

            self.nnet = self._build_nnet(in_channels, ydim, idim, nblocks)

    def _build_nnet(self, in_channels, ydim, idim, nblocks):
        nnet_list = []
        nnet_list.append(nn.Conv2d(in_channels, ydim, (3,3), stride=1, padding=1))
        for i in range(nblocks):
            nnet_list.append(Bottleneck(ydim, ydim, idim))
        return layers.ExtendedSequential(*nnet_list)

    def forward(self, x, u):
        return self.nnet(x), u

    def build_clone(self):
        nnet_list = []
        for nnet in self.nnet:
            nnet_list.append(clone_primitive(nnet))
        nnet = nn.Sequential(*nnet_list)
        return VarDeqConditioningLayer(None, None, None, None, nnet)

    def build_jvp_net(self, x, u):
        with torch.no_grad():
            return nn.Identity(), self.nnet(x).detach().clone(), u


class VarDeqLipschitzDenseLayer(nn.Module):
    def __init__(self, lipschitz_dense_layer, in_channels, densenet_growth, ydim, idim, *args):
        super(VarDeqLipschitzDenseLayer, self).__init__()

        if in_channels is None:
            # constructor called from build_clone
            self.lipschitz_dense_layer = lipschitz_dense_layer
            self.inject_layer = args[0]
        else:
            self.lipschitz_dense_layer = lipschitz_dense_layer
            self.inject_layer = Bottleneck(ydim, densenet_growth, idim)

    def forward(self, x, u):
        u, v = self.lipschitz_dense_layer.forward(u, concat=False)
        v = v + self.inject_layer(x)
        return x, torch.cat([u,v], dim=1)

    def build_clone(self):
        return VarDeqLipschitzDenseLayer(self.lipschitz_dense_layer.build_clone(), None, None, None, None, self.inject_layer.build_clone())

    def build_jvp_net(self, x, u):
        with torch.no_grad():
            jvp_net, u, v = self.lipschitz_dense_layer.build_jvp_net(u, concat=False)
            v = v + self.inject_layer(x)
            return jvp_net, x, torch.cat([u,v], dim=1)


class VarDeqWrapper(nn.Module):
    def __init__(self, nnet):
        super(VarDeqWrapper, self).__init__()

        self.nnet = nnet

    def forward(self, x, u):
        return x, self.nnet(u)

    def build_clone(self):
        return VarDeqWrapper(self.nnet.build_clone())

    def build_jvp_net(self, x, u):
        jvp_net, v = self.nnet.build_jvp_net(u)
        return jvp_net, x, v


class VarDeqBlock(nn.Module):
    def __init__(self,
        nnet, 
        geom_p=0.5,
        lamb=2.,
        n_power_series=None,
        exact_trace=False,
        brute_force=False,
        n_samples=1,
        n_exact_terms=2,
        n_dist='geometric',
        neumann_grad=True,
        grad_in_forward=False,
        monotone_resolvent=False):

        super(VarDeqBlock, self).__init__()
        self.nnet = nnet
        self.n_dist = n_dist
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1. - geom_p)), requires_grad=False)
        self.lamb = nn.Parameter(torch.tensor(lamb), requires_grad=False)
        self.n_samples = n_samples
        self.n_power_series = n_power_series
        self.exact_trace = exact_trace
        self.brute_force = brute_force
        self.n_exact_terms = n_exact_terms
        self.grad_in_forward = grad_in_forward
        self.neumann_grad = neumann_grad
        self.monotone_resolvent = monotone_resolvent

        # store the samples of n.
        self.register_buffer('last_n_samples', torch.zeros(self.n_samples))
        self.register_buffer('last_firmom', torch.zeros(1))
        self.register_buffer('last_secmom', torch.zeros(1))

    def forward(self, x, u, logpx=None):
        class XWrapper(nn.Module):
            def __init__(self, nnet, x):
                super(XWrapper, self).__init__()
                self.nnet = nnet
                self.x = x
            def forward(self, u):
                x = self.x
                v = u
                for m in self.nnet:
                    x, v = m(x, v)
                return v
            def build_jvp_net(self, u):
                jvp_net, x, v = self.nnet.build_jvp_net(self.x, u)
                return jvp_net, v
        xwrapper = XWrapper(self.nnet, x)

        if self.monotone_resolvent:
            xwrapper_copy = XWrapper(self.nnet.build_clone(), x.detach().clone())
            u0 = u.clone().detach()
            w_value = solvers.RootFind.apply(lambda z: xwrapper_copy(z), math.sqrt(2)*u0, 'banach', 1e-6, 2000).detach()

            w_proxy = math.sqrt(2)*u0 - xwrapper(w_value) # For backwarding to parameters in func
            w = solvers.MonotoneBlockBackward.apply(lambda z: xwrapper_copy(z), w_proxy, math.sqrt(2)*u, 'banach', 1e-9, 100)
            v = math.sqrt(2)*w - u

            if logpx is None:
                return x, v
            else:
                return x, v, logpx - self._logdetgrad_monotone_resolvent(w, xwrapper)

        else:
            if logpx is None:
                v = u + xwrapper(u)
                return x, v
            else:
                g, logdetgrad = self._logdetgrad(u, xwrapper)
                return x, u + g, logpx - logdetgrad

    def _logdetgrad(self, x, xwrapper):
        """Returns g(x) and logdet|d(x+g(x))/dx|."""

        with torch.enable_grad():
            if (self.brute_force or not self.training) and (x.ndimension() == 2 and x.shape[1] == 2):
                ###########################################
                # Brute-force compute Jacobian determinant.
                ###########################################
                x = x.requires_grad_(True)
                g = xwrapper(x)
                # Brute-force logdet only available for 2D.
                jac = batch_jacobian(g, x)
                batch_dets = (jac[:, 0, 0] + 1) * (jac[:, 1, 1] + 1) - jac[:, 0, 1] * jac[:, 1, 0]
                return g, torch.log(torch.abs(batch_dets)).view(-1, 1)

            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn = lambda m: poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)

            if self.training:
                if self.n_power_series is None:
                    # Unbiased estimation.
                    lamb = self.lamb.item()
                    n_samples = sample_fn(self.n_samples)
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * \
                        sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
                else:
                    # Truncated estimation.
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.
            else:
                # Unbiased estimation with more exact terms.
                lamb = self.lamb.item()
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + 20
                coeff_fn = lambda k: 1 / rcdf_fn(k, 20) * \
                    sum(n_samples >= k - 20) / len(n_samples)

            if not self.exact_trace:
                ####################################
                # Power series with trace estimator.
                ####################################
                vareps = torch.randn_like(x)

                # Choose the type of estimator.
                if self.training and self.neumann_grad:
                    estimator_fn = neumann_logdet_estimator
                else:
                    estimator_fn = basic_logdet_estimator

                # Do backprop-in-forward to save memory.
                power_series_coeff_fn = lambda k: (-1)**(k + 1) / k
                if self.training and self.grad_in_forward:
                    g, logdetgrad = mem_eff_wrapper(
                        estimator_fn, xwrapper, x, n_power_series, power_series_coeff_fn, vareps, coeff_fn, self.training
                    )
                else:
                    x = x.requires_grad_(True)
                    g, logdetgrad = estimator_fn(xwrapper, x, n_power_series, power_series_coeff_fn, vareps, coeff_fn, self.training)
            else:
                ############################################
                # Power series with exact trace computation.
                ############################################
                x = x.requires_grad_(True)
                g = xwrapper(x)
                jac = batch_jacobian(g, x)
                logdetgrad = batch_trace(jac)
                jac_k = jac
                for k in range(2, n_power_series + 1):
                    jac_k = torch.bmm(jac, jac_k)
                    logdetgrad = logdetgrad + coeff_fn(k) * batch_trace(jac_k)

            if self.training and self.n_power_series is None:
                self.last_n_samples.copy_(torch.tensor(n_samples).to(self.last_n_samples))
                estimator = logdetgrad.detach()
                self.last_firmom.copy_(torch.mean(estimator).to(self.last_firmom))
                self.last_secmom.copy_(torch.mean(estimator**2).to(self.last_secmom))
            return g, logdetgrad.view(-1, 1)

    def _logdetgrad_monotone_resolvent(self, w, xwrapper):
        """Returns logdet|d(sqrt(2)*(Id+g)^{-1}(sqrt(2)*x))/dx|."""

        with torch.enable_grad():
            if (self.brute_force or not self.training) and (w.ndimension() == 2 and w.shape[1] == 2):
                ###########################################
                # Brute-force compute Jacobian determinant.
                ###########################################
                w = w.requires_grad_(True)
                g = xwrapper(w)
                # Brute-force logdet only available for 2D.
                jac = batch_jacobian(g, w)
                batch_dets1 = (1 - jac[:, 0, 0]) * (1 - jac[:, 1, 1]) - jac[:, 0, 1] * jac[:, 1, 0]
                batch_dets2 = (1 + jac[:, 0, 0]) * (1 + jac[:, 1, 1]) - jac[:, 0, 1] * jac[:, 1, 0]
                return (torch.log(torch.abs(batch_dets1)) - torch.log(torch.abs(batch_dets2))).view(-1, 1)

            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: iresblock.geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: iresblock.geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn = lambda m: iresblock.poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: iresblock.poisson_1mcdf(lamb, k, offset)

            if self.training:
                if self.n_power_series is None:
                    # Unbiased estimation.
                    lamb = self.lamb.item()
                    n_samples = sample_fn(self.n_samples)
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * \
                        sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
                else:
                    # Truncated estimation.
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.
            else:
                # Unbiased estimation with more exact terms.
                lamb = self.lamb.item()
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + 20
                coeff_fn = lambda k: 1 / rcdf_fn(k, 20) * \
                    sum(n_samples >= k - 20) / len(n_samples)

            if not self.exact_trace:
                ####################################
                # Power series with trace estimator.
                ####################################
                vareps = torch.randn_like(w)

                # Choose the type of estimator.
                if self.training and self.neumann_grad:
                    estimator_fn = iresblock.neumann_logdet_estimator
                else:
                    estimator_fn = iresblock.basic_logdet_estimator

                # Do backprop-in-forward to save memory.
                power_series_coeff_fn = lambda k: (-2)/k if k%2 == 1 else 0
                if self.training and self.grad_in_forward:
                    g, logdetgrad = iresblock.mem_eff_wrapper(
                        estimator_fn, xwrapper, w, n_power_series, power_series_coeff_fn, vareps, coeff_fn, self.training
                    )
                else:
                    w = w.requires_grad_(True)
                    g, logdetgrad = estimator_fn(xwrapper, w, n_power_series, power_series_coeff_fn, vareps, coeff_fn, self.training)
            else:
                raise NotImplementedError()
                '''
                ############################################
                # Power series with exact trace computation.
                ############################################
                x = x.requires_grad_(True)
                g = self.nnet(x)
                jac = batch_jacobian(g, x)
                logdetgrad = batch_trace(jac)
                jac_k = jac
                for k in range(2, n_power_series + 1):
                    jac_k = torch.bmm(jac, jac_k)
                    logdetgrad = logdetgrad + coeff_fn(k) * batch_trace(jac_k)
                '''

            if self.training and self.n_power_series is None:
                self.last_n_samples.copy_(torch.tensor(n_samples).to(self.last_n_samples))
                estimator = logdetgrad.detach()
                self.last_firmom.copy_(torch.mean(estimator).to(self.last_firmom))
                self.last_secmom.copy_(torch.mean(estimator**2).to(self.last_secmom))
            return logdetgrad.view(-1, 1)


class VarDeq(nn.Module):
    def __init__(self, nnet, nbits):
        super(VarDeq, self).__init__()
        self.nnet = nnet
        self.nbits = nbits
        self.nbins = 2 ** nbits

    def forward(self, x, logpx=None):
        u = torch.randn_like(x)
        eps_nll = 0.5 * (u ** 2 + math.log(2 * math.pi))

        for m in self.nnet:
            if logpx is not None:
                _, u, logpx = m.forward(x, u, logpx)
            else:
                _, u = m.forward(x, u)
        u = torch.sigmoid(u)
        x = (x * (self.nbins - 1) + u) / self.nbins

        if logpx is not None:
            sigmoid_ldj = safe_log(u) + safe_log(1. - u)
            logpx = logpx - (eps_nll + sigmoid_ldj).flatten(1).sum(-1)
            return x, logpx
        else:
            return x
