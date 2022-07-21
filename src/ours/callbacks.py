import einops
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from torchvision.utils import make_grid
import sys
sys.path.append('../../')
from src.ours.sparsemax_k import sparsemax_k


class ReconstructionCallback(pl.Callback):
    def __init__(self, dataset, every_n_batches=100) -> None:
        super().__init__()
        self.every_n_batches = every_n_batches
        
        obs, actions, vp, nonterms, dropped, player_pos = dataset.get_no_drop(0)
        self.obs = torch.from_numpy(obs[None])
        self.actions = torch.from_numpy(actions[None])
        if player_pos is not None:
            self.player_pos = torch.from_numpy(player_pos[None])
        else:
            self.player_pos = None
        self.nonterms = torch.from_numpy(nonterms[None])
        self.value_prefixes = torch.from_numpy(vp[None])
        self.dropped = torch.from_numpy(dropped[None])
        self.mu = dataset.mu
        self.sigma = dataset.sigma
        
    
    def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (pl_module.global_step+1) % self.every_n_batches == 0:
            self.visualize_recon(trainer, pl_module)
    
    def visualize_recon(self, trainer, pl_module):
        if self.obs.device != pl_module.device:
            self.obs = self.obs.to(pl_module.device)
            self.actions = self.actions.to(pl_module.device)
            self.dropped = self.dropped.to(pl_module.device)
            if self.player_pos is not None:
                self.player_pos = self.player_pos.to(pl_module.device)
            self.nonterms = self.nonterms.to(pl_module.device)
            self.value_prefixes = self.value_prefixes.to(pl_module.device)

        outputs = pl_module(self.obs, self.actions, self.value_prefixes, self.nonterms, self.dropped, self.player_pos)
        posterior_belief_sequence = outputs['posterior_belief_sequence']
        posterior_bit_vec_sequence = outputs['posterior_bit_vec_sequence']
        
        obs_hat = pl_module.emission.decode_only(posterior_belief_sequence, posterior_bit_vec_sequence).to('cpu').float()[0]
        num_views = self.obs.shape[2]
        images = torch.stack(
            [(self.obs[0,:,i].to('cpu')*self.sigma)+self.mu for i in range(num_views)]\
            + [(obs_hat[:,i].to('cpu')*self.sigma)+self.mu for i in range(num_views)],
            dim = 1
        ).reshape(2*num_views*obs_hat.shape[0], 1, obs_hat.shape[2], obs_hat.shape[3])
        
        # log images
        pl_module.logger.experiment.log({'Reconstruction': wandb.Image(make_grid(images, nrow=2*num_views))})
        
class ExtrapolateCallback(pl.Callback):
    def __init__(self, dataset=None, save_to_disk=False, every_n_batches=100):
        """
        Inputs:
            batch_size - Number of images to generate
            dataset - Dataset to sample from
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.every_n_batches = every_n_batches
        self.save_to_disk = save_to_disk
        self.initial_loading = False
        obs, actions, vp, terms, dropped, player_pos = dataset.get_no_drop(0)
        self.obs = torch.from_numpy(obs)
        self.actions = torch.from_numpy(actions[None])
        self.dropped = torch.from_numpy(dropped[None])
        self.mu = dataset.mu
        self.sigma = dataset.sigma

        self.obs_logits = None

    def on_batch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (pl_module.global_step+1) % self.every_n_batches == 0:
            self.extrapolate(trainer, pl_module)
    
    # def on_epoch_end(self, trainer, pl_module):
    #     """
    #     This function is called after every epoch.
    #     Call the save_and_sample function every N epochs.
    #     """
    #     self.extrapolate(trainer, pl_module, pl_module.global_step+1)

    @torch.no_grad()
    def extrapolate(self, trainer, pl_module):
        if self.obs.device != pl_module.device:
            self.obs = self.obs.to(pl_module.device)
            self.actions = self.actions.to(pl_module.device)
            self.dropped = self.dropped.to(pl_module.device)
        
        self.prior = pl_module.prior(1) # batch size 1
        if pl_module.hparams.sparsemax:
            self.prior, self.state_idcs = sparsemax_k(self.prior, pl_module.hparams.sparsemax_k) 
        else:
            state_logits = F.log_softmax(self.prior, dim=-1)
            temp = state_logits[:,0]
            for i in range(1, state_logits.shape[1]):
                temp = temp[...,None] + state_logits[:,i,None,:]
                temp = torch.flatten(temp, start_dim=1)
            self.prior = F.softmax(temp, dim=-1)
            self.state_idcs = None
        
        # extrapolations
        ent = -(self.prior * self.prior.log())
        ent[self.prior == 0] = 0
        ent = ent.sum()
        print(f"\n0: prior entropy: {ent:.4f}")
        self.posterior_0, _ = pl_module.network.compute_posterior(self.prior, self.state_idcs, self.obs[0, None], self.dropped[:,0])
        
        ent = -(self.posterior_0 * self.posterior_0.log())
        ent[self.posterior_0 == 0] = 0
        ent = ent.sum()
        print(f"0: post  entropy: {ent:.4f}")
        state_belief_prior_sequence, state_bit_vec_sequence = pl_module.network.k_step_extrapolation(self.posterior_0, self.state_idcs, self.actions, self.actions.shape[1])
        state_belief_prior_sequence = torch.cat([self.posterior_0[:,None], state_belief_prior_sequence], dim=1)
        if state_bit_vec_sequence is not None:
            state_bit_vec_sequence = torch.cat([self.state_idcs[None], state_bit_vec_sequence], dim=1)
        
        for i in range(1,state_belief_prior_sequence.shape[1]):
            ent = -(state_belief_prior_sequence[:,i] * state_belief_prior_sequence[:,i].log())
            ent[state_belief_prior_sequence[:,i] == 0] = 0
            ent = ent.sum()
            print(f"{i}: prior entropy: {ent:.4f}")
        
        obs_hat = pl_module.emission.decode_only(state_belief_prior_sequence, state_bit_vec_sequence).to('cpu').float()[0]
        num_views = self.obs.shape[1]
        images = torch.stack(
            [(self.obs[:,i].to('cpu')*self.sigma)+self.mu for i in range(num_views)]\
            + [(obs_hat[:-1,i].to('cpu')*self.sigma)+self.mu for i in range(num_views)],
            dim = 1
        ).reshape(2*num_views*self.obs.shape[0], 1, self.obs.shape[2], self.obs.shape[3])
        # log images
        pl_module.logger.experiment.log({'Extrapolation': wandb.Image(make_grid(images, nrow=2*num_views))})
