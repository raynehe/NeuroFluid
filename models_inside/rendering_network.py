import torch
import torch.nn as nn
from models.embedder import *
import numpy as np
from pytorch3d.ops import ball_query
from models.nerf import Embedding

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            position_encoding_size,
            dir_encoding_size,
            normal_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            NN_search = {},
            encoding = {},
            bias=1.0,
            geometric_init=True
    ):
        super().__init__()

        self.mode = mode
        # dims = [d_in + feature_vector_size] + dims + [d_out]
        dims = [d_in + feature_vector_size + position_encoding_size + dir_encoding_size + normal_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)
        self.radius = NN_search['search_raduis_scale'] * NN_search['particle_radius']
        self.fix_radius = NN_search['fix_radius']
        self.num_neighbor = NN_search['N_neighbor']
        self.encoding_cfg = encoding
        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)
        if self.encoding_cfg['density']:
            self.embedding_density = Embedding(1, 4)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.normal_(lin.weight, mean=0.0, std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires_view > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires_view > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(negative_slope=1e-2)

    def forward(self, points, normals, view_dirs, feature_vectors, physical_particles, rays, ro):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
        # add position embdding here
        # search
        dists_0, indices_0, neighbors_0, radius_0 = self.search(points, physical_particles, self.fix_radius)
        # embedding attributes
        pos_like_feats_0, dirs_like_feats_0, num_nn_0 = self.embedding_local_geometry(dists_0, indices_0, neighbors_0, radius_0, points, rays, ro)
        pos_input_feats_0 = torch.cat(pos_like_feats_0, dim=1) # 100352, 198
        dir_input_feats_0 = torch.cat(dirs_like_feats_0, dim=1) # 100352, 54

        if self.mode == 'idr':
            rendering_input = torch.cat([pos_input_feats_0, dir_input_feats_0, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([dir_input_feats_0, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                # x = self.leakyrelu(x)

        x = self.sigmoid(x)
        return x

    def search(self, ray_particles, particles, fix_radius):
        raw_data = particles.unsqueeze(0).repeat(ray_particles.shape[0], 1, 1)
        if fix_radius:
            radius = self.radius
            dists, indices, neighbors = ball_query(p1=ray_particles, 
                                                   p2=raw_data, 
                                                   radius=radius, K=self.num_neighbor)
        # else:
        #     radius = self.get_search_radius(self.radius, ray_particles[:,:,-1] - ro[-1], focal)
        #     dists, indices, neighbors = self._ball_query(ray_particles, raw_data, radius, self.num_neighbor)
        return dists, indices, neighbors, radius
    
    def embedding_local_geometry(self, dists, indices, neighbors, radius, ray_particles, rays = [], ro = [], sigma_only=False):
        """
        pos like feats
            1. smoothed positions
            2. ref hit pos, i.e., ray position
            3. density
            3. variance
        dir like feats
            1. hit direction, i.e., ray direction
            2. main direction after PCA
        """
        # calculate mask
        nn_mask = dists.ne(0)
        num_nn = nn_mask.sum(-1, keepdim=True)

        # hit pos and hit direction (basic in NeRF formulation)
        pos_like_feats = []
        hit_pos = ray_particles.reshape(-1,3)
        hit_pos_embedded = self.embedding_xyz(hit_pos)
        pos_like_feats.append(hit_pos_embedded)
        if not sigma_only:
            hit_dir = rays[:,3:]
            hit_dir_embedded = self.embedding_dir(hit_dir)
            hit_dir_embedded = torch.repeat_interleave(hit_dir_embedded, repeats=ray_particles.shape[1], dim=0)
            dir_like_feats = []
            dir_like_feats.append(hit_dir_embedded)
        # smoothing 
        smoothed_pos, density = self.smoothing_position(ray_particles, neighbors, radius, num_nn, exclude_ray=self.encoding_cfg['exclude_ray'])
        if not sigma_only:
            smoothed_dir = self.get_particles_direction(smoothed_pos.reshape(-1, 3), ro)
        # density
        if self.encoding_cfg['density']:
            density_embedded = self.embedding_density(density.reshape(-1, 1))
            pos_like_feats.append(density_embedded)
        # smoothed pos
        if self.encoding_cfg['smoothed_pos']:
            smoothed_pos_embedded = self.embedding_xyz(smoothed_pos.reshape(-1, 3))
            pos_like_feats.append(smoothed_pos_embedded)
        # variance
        if self.encoding_cfg['var']:
            vec_pp2rp = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, 3).to(neighbors.device)
            vec_pp2rp[nn_mask] = (neighbors - ray_particles.unsqueeze(-2))[nn_mask]
            vec_pp2rp_mean = vec_pp2rp.sum(-2) / (num_nn+1e-12)
            variance = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, 3).to(neighbors.device)
            variance[nn_mask] = ((vec_pp2rp - vec_pp2rp_mean.unsqueeze(-2))**2)[nn_mask]
            variance = variance.sum(-2) / (num_nn+1e-12)
            variance_embedded = self.embedding_xyz(variance.reshape(-1,3))
            pos_like_feats.append(variance_embedded)
        # smoothed dir
        if not sigma_only:
            if self.encoding_cfg['smoothed_dir']:
                smoothed_dir_embedded = self.embedding_dir(smoothed_dir)
                dir_like_feats.append(smoothed_dir_embedded)
        if not sigma_only:
            return pos_like_feats, dir_like_feats, num_nn
        else:
            return pos_like_feats
    
    def smoothing_position(self, ray_pos, nn_poses, raduis, num_nn, exclude_ray=True, larger_alpha=0.9, smaller_alpha=0.1):
        dists = torch.norm(nn_poses - ray_pos.unsqueeze(-2), dim=-1)
        weights = torch.clamp(1 - (dists / raduis) ** 3, min=0)
        weighted_nn = (weights.unsqueeze(-1) * nn_poses).sum(-2) / (weights.sum(-1, keepdim=True)+1e-12)
        if exclude_ray:
            pos = weighted_nn
        else:
            if self.encoding_cfg['same_smooth_factor']:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
            else:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
                alpha[num_nn.le(20)] = smaller_alpha
            pos = ray_pos * (1-alpha) + weighted_nn * alpha
        return pos, weights.sum(-1, keepdim=True)
    
    def get_particles_direction(self, particles, ro):
        ros = ro.expand(particles.shape[0], -1)
        dirs = particles - ros
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        return dirs
