import argparse

def create_train_parser():
    parser = argparse.ArgumentParser()
    
    # env settings
    parser.add_argument('--num_seeds', type=int, default=None)
    parser.add_argument('--null_value', type=int, default=1)
    parser.add_argument('--num_views', type=int, default=1)
    parser.add_argument('--percentage', type=float, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--test_only_dropout', default='no', type=str, choices=['yes', 'no'])
    parser.add_argument('--max_datapoints', type=int, default=None)
    parser.add_argument('--obs_scale', type=int, default=1)
    parser.add_argument('--num_actions', type=int, default=4)
    
    ## model args
    parser.add_argument('--kl_balancing_coeff', type=float, default=0.8)
    parser.add_argument('--kl_scaling', type=float, default=0.1)
    parser.add_argument('--l_unroll', type=int, default=1)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--num_variables', type=int, default=10)
    parser.add_argument('--codebook_size', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--disable_recon_loss', default='no', type=str, choices=['yes', 'no'])
    parser.add_argument('--sparsemax', default='no', type=str, choices=['yes', 'no'])
    parser.add_argument('--disable_vp', default='no', type=str, choices=['yes', 'no'])
    parser.add_argument('--sparsemax_k', type=int, default=30)
    parser.add_argument('--action_layer_dims', type=int, nargs='*', default=None)
    parser.add_argument('--vp_layer_dims', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--vp_batchnorm', type=str, choices=['yes', 'no'], default='no')
    parser.add_argument('--force_uniform_prior', type=str, choices=['yes', 'no'], default='no')
    parser.add_argument('--prior_noise_scale', type=float, default=0.0)
    parser.add_argument('--kernel_size', type=int, default=3, help="Size of the conv kernel")
    parser.add_argument('--depth', type=int, default=16, help="Scaling parameter for conv net")
    
    # training args
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.000001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=float, default=10)
    parser.add_argument('--gradient_clip_val', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--track_grad_norm', type=int, default=-1)
    parser.add_argument('--max_len', type=int, default=10, help='Max length of an episode for batching purposes. Rest will be padded.')
    
    return parser