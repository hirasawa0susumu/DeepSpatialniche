import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
from collections import OrderedDict

from .transport import create_transport, Sampler

class DeepSpatialModule(pl.LightningModule):
    """
    DeepSpatialniche Module for Training & Inference.

    Parameters
    ----------
    args : dict or argparse.Namespace
        Configuration dictionary containing hyperparameters such as learning rate, 
        path type, and sampling settings.
    model : torch.nn.Module
        The core neural network architecture (e.g., the GiT model) that predicts 
        the velocity fields.
    """
    def __init__(self, args, model, niche_encoder=None):
        """
        Initializes an LightningModule instance for DeepSpatialniche.
        """
        super().__init__()
        self.save_hyperparameters(args)

        # Core Model
        self.model = model
        self.niche_encoder = niche_encoder

        # EMA Setup
        self.ema_decay = self.hparams.get('ema_decay', 0.999)
        self.ema_model = deepcopy(self.model)
        if niche_encoder is not None:
            self.ema_niche_encoder = deepcopy(niche_encoder)
            self._freeze(self.ema_niche_encoder)
        else:
            self.ema_niche_encoder = None
        self._freeze(self.ema_model)

        # Transport & Path Setup
        self.transport = create_transport(
            path_type=self.hparams.path_type,
            prediction=self.hparams.prediction,
            train_eps=self.hparams.get('train_eps', 0.02),
            sample_eps=self.hparams.get('sample_eps', 0.02),
        )
        self.sampler = Sampler(self.transport)

    def _freeze(self, module):
        """Freeze model parameters for EMA or evaluation."""
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def configure_optimizers(self):
        """Initialize AdamW optimizer with weight decay."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.get('weight_decay', 1e-5)
        )

    # ============================================================
    # Training & Validation Logic
    # ============================================================
    def _shared_step(self, batch):
        """
        Computes joint Flow Matching losses across spatial and molecular dimensions.
        """
        x0, x1 = batch['x0'], batch['x1']
        g0, g1 = batch['g0'], batch['g1']
        c0, c1 = batch['c0'], batch['c1']
        z0, z1 = batch['z0'], batch['z1']
        delta_z = batch['delta_z']

        # Sample time steps
        t, _, _ = self.transport.sample(x1)

        # Plan paths (interpolation)
        _, xt, ux_t = self.transport.path_sampler.plan(t, x0, x1)
        _, gt, ug_t = self.transport.path_sampler.plan(t, g0, g1)
        _, ct, uc_t = self.transport.path_sampler.plan(t, c0, c1)
        _, zt, _ = self.transport.path_sampler.plan(t, z0, z1)

        # Multi-scale niche with dynamic interpolation at training time t
        niche_token = None
        if self.niche_encoder is not None:
            niche_dropout = self.hparams.get('niche_dropout', 0.3)
            is_multiscale = 'g_nbr_local' in batch
            if self.training and torch.rand(1).item() < niche_dropout:
                niche_token = None
            elif is_multiscale:
                has_dynamic = 'g_nbr_target_local' in batch
                t_3d = t[:, None, None, None]

                def _interp(scale):
                    gs = batch[f'g_nbr_{scale}']
                    ds = batch[f'delta_nbr_{scale}']
                    mk = batch.get(f'mask_nbr_{scale}')
                    if has_dynamic:
                        gtgt = batch[f'g_nbr_target_{scale}']
                        ptgt = batch[f'pos_nbr_target_{scale}']
                        gi = (1 - t_3d) * gs + t_3d * gtgt
                        pa = x0.unsqueeze(1) + ds
                        pi = (1 - t_3d) * pa + t_3d * ptgt
                        di = pi - xt.unsqueeze(1)
                        dst = di.norm(dim=-1)
                    else:
                        gi = gs
                        pa = x0.unsqueeze(1) + ds
                        di = pa - xt.unsqueeze(1)
                        dst = di.norm(dim=-1)
                    return gi, di, dst, mk

                gl, dl, dstl, ml = _interp('local')
                gm_, dm_, dstm, mm = _interp('mid')
                gg, dg, dstg, mg = _interp('global')

                niche_token = self.niche_encoder(
                    g_center=gt, pos_center=xt,
                    g_nbrs_local=gl, delta_local=dl,
                    dist_local=dstl, mask_local=ml,
                    g_nbrs_mid=gm_, delta_mid=dm_,
                    dist_mid=dstm, mask_mid=mm,
                    g_nbrs_global=gg, delta_global=dg,
                    dist_global=dstg, mask_global=mg,
                )
            elif 'g_nbr' in batch:
                niche_token = self.niche_encoder(
                    g_center=g0, pos_center=x0,
                    g_nbrs=batch['g_nbr'],
                    delta_nbrs=batch['delta_nbr'],
                    dist_nbrs=batch['dist_nbr'],
                    mask_nbr=batch.get('mask_nbr'),
                )

        # 3. Predict velocity fields
        vx_pred, vg_pred, vc_pred = self.model(
            xt=xt, gt=gt, t=t, zt=zt, delta_z=delta_z, ct=ct,
            niche_token=niche_token,
        )

        # Compute losses (Mean Squared Error on velocity)
        # Spatial loss (X, Y)
        loss_x = self.transport.loss_fn(vx_pred, x0, xt, t, ux_t).mean()
        # Gene loss
        loss_g = self.transport.loss_fn(vg_pred, g0, gt, t, ug_t).mean()
        # Cell type loss (on one-hot/continuous space)
        loss_c = self.transport.loss_fn(vc_pred, c0, ct, t, uc_t).mean()

        # Weighted total loss
        lambda_g = self.hparams.get('lambda_g', 0.1)
        lambda_c = self.hparams.get('lambda_c', 10.0)
        loss_total = loss_x + (lambda_g * loss_g) + (lambda_c * loss_c)
        
        return {
            'loss': loss_total,
            'loss_x': loss_x,
            'loss_g': loss_g,
            'loss_c': loss_c
        }

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('loss', loss['loss'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('loss_x', loss['loss_x'], on_epoch=True)
        self.log('loss_g', loss['loss_g'], on_epoch=True)
        self.log('loss_c', loss['loss_c'], on_epoch=True)
        return loss
    
    # ============================================================
    # EMA Update Logic
    # ============================================================
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA parameters after each optimizer step."""
        self._update_ema()

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA model weights with exponential decay."""
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)
        if self.ema_niche_encoder is not None and self.niche_encoder is not None:
            for ema_param, model_param in zip(
                self.ema_niche_encoder.parameters(), self.niche_encoder.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

    def on_load_checkpoint(self, checkpoint):
        """Ensure EMA model is also loaded from checkpoint."""
        self._update_ema() # Warm start EMA with current loaded model params

    # ============================================================
    # Inference / Sampling Logic
    # ============================================================
    @torch.no_grad()
    def sample(self, batch, mode="ODE", steps=20):
        """ODE integration with multi-scale dynamic niche refresh."""
        self.ema_model.eval()

        sample_config = {
            'num_steps': steps,
            'sampling_method': self.hparams.get('sampling_method', 'dopri5'),
            'atol': self.hparams.get('atol', 1e-5),
            'rtol': self.hparams.get('rtol', 1e-5)
        }
        sampler_fn = (self.sampler.sample_ode(**sample_config)
                      if mode == "ODE" else self.sampler.sample_sde(**sample_config))

        x0, g0, c0 = batch['x0'], batch['g0'], batch['c0']
        x_dim, g_dim = x0.shape[-1], g0.shape[-1]
        z0, z1, delta_z = batch['z0'], batch['z1'], batch['delta_z']
        niche_token = batch.get('niche_token', None)
        niche_nbr_data = batch.get('niche_nbr_data', None)
        niche_refresh_every = self.hparams.get('niche_refresh_steps', 5)

        _niche_enc = (self.ema_niche_encoder if getattr(self, 'ema_niche_encoder', None) is not None
                      else self.niche_encoder)

        _niche = [niche_token]
        _step_counter = [0]

        init_state = torch.cat([x0, g0, c0], dim=-1)

        def velocity_field_wrapper(joint_state_t, t):
            xt = joint_state_t[..., :x_dim]
            gt = joint_state_t[..., x_dim : x_dim + g_dim]
            ct = joint_state_t[..., x_dim + g_dim :]

            t_val = (t.item() if (torch.is_tensor(t) and t.dim() == 0)
                     else (t[0].item() if torch.is_tensor(t) else t))
            t_tensor = torch.full((xt.shape[0],), t_val, device=xt.device, dtype=xt.dtype)
            _, zt, _ = self.transport.path_sampler.plan(t_tensor, z0, z1)

            # Periodic multi-scale dynamic niche refresh
            if niche_nbr_data is not None and _step_counter[0] % niche_refresh_every == 0:
                has_dyn = 'g_nbr_target_local' in niche_nbr_data
                is_ms = 'g_nbr_local' in niche_nbr_data
                tc = min(max(t_val, 0.0), 1.0)

                if is_ms:
                    def _ref(scale):
                        gs = niche_nbr_data[f'g_nbr_{scale}']
                        ds = niche_nbr_data[f'delta_nbr_{scale}']
                        mk = niche_nbr_data.get(f'mask_nbr_{scale}')
                        if has_dyn:
                            gtgt = niche_nbr_data[f'g_nbr_target_{scale}']
                            ptgt = niche_nbr_data[f'pos_nbr_target_{scale}']
                            gi = (1 - tc) * gs + tc * gtgt
                            pa = x0.unsqueeze(1) + ds
                            pi = (1 - tc) * pa + tc * ptgt
                            di = pi - xt.unsqueeze(1)
                            dst = di.norm(dim=-1)
                        else:
                            gi = gs
                            di = (x0.unsqueeze(1) + ds) - xt.unsqueeze(1)
                            dst = di.norm(dim=-1)
                        return gi, di, dst, mk

                    gl, dl, dstl, ml = _ref('local')
                    gm_, dm_, dstm, mm = _ref('mid')
                    gg, dg, dstg, mg = _ref('global')

                    _niche[0] = _niche_enc(
                        g_center=gt, pos_center=xt,
                        g_nbrs_local=gl, delta_local=dl,
                        dist_local=dstl, mask_local=ml,
                        g_nbrs_mid=gm_, delta_mid=dm_,
                        dist_mid=dstm, mask_mid=mm,
                        g_nbrs_global=gg, delta_global=dg,
                        dist_global=dstg, mask_global=mg,
                    )
            _step_counter[0] += 1

            vx, vg, vc = self.ema_model(
                xt=xt, gt=gt, t=t_tensor, zt=zt, delta_z=delta_z, ct=ct,
                niche_token=_niche[0],
            )
            return torch.cat([vx, vg, vc], dim=-1)

        trajectory = sampler_fn(init_state, velocity_field_wrapper)
        if isinstance(trajectory, (list, tuple)):
            trajectory = torch.stack(trajectory, dim=0)

        return {
            'x_traj': trajectory[..., :x_dim],
            'g_traj': trajectory[..., x_dim : x_dim + g_dim],
            'c_traj_discrete': torch.argmax(trajectory[..., x_dim + g_dim:], dim=-1)
        }