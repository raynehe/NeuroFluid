import torch
import torch.nn as nn
from models.embedder import *
import numpy as np
from pytorch3d.ops import ball_query
from models.nerf import Embedding

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            position_encoding_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            NN_search = {},
            encoding = {},
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in + position_encoding_size] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        # dims [198, 256, 256, 256, 256, 256, 256, 256, 256, 257]

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.radius = NN_search['search_raduis_scale'] * NN_search['particle_radius']
        self.fix_radius = NN_search['fix_radius']
        self.num_neighbor = NN_search['N_neighbor']
        self.encoding_cfg = encoding
        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)
        if self.encoding_cfg['density']:
            self.embedding_density = Embedding(1, 4)

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - d_in
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    # torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.normal_(lin.weight, mean=0.0, std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.leakyrelu = nn.LeakyReLU(negative_slope=1e-2)
        # self.BN = nn.BatchNorm1d(256)
        # self.BN_skip = nn.BatchNorm1d(256-198)
        # self.BN_final = nn.BatchNorm1d(257)

    def forward(self, input, physical_particles):
        if self.embed_fn is not None:
            input = self.embed_fn(input)
        # add position embdding here
        # search
        dists, indices, neighbors, radius = self.search(input, physical_particles, self.fix_radius)
        # embedding attributes
        pos_like_feats_0, num_nn = self.embedding_local_geometry(dists, indices, neighbors, radius, input)
        pos_input_feats_0 = torch.cat(pos_like_feats_0, dim=1) # 512*128=65536, 198
        input = input.reshape(-1, 3) # 65536, 3

        x = pos_input_feats_0 # 65536, 198

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
                # x = self.leakyrelu(x)
                # if (l+1) in self.skip_in:
                #     x = self.BN_skip(x)
                # else:
                #     x = self.BN(x)
        # x = self.BN_final(x)

        return x, dists, num_nn # 66536, 257

    def gradient(self, x, physical_particles, iseval=False):
        x.requires_grad_(True)
        if (iseval):
            torch.set_grad_enabled(True)
        y = self.forward(x, physical_particles)[0][:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        if (iseval):
            torch.set_grad_enabled(False)
            gradients = gradients.detach()
        return gradients

    def get_outputs(self, x, physical_particles, iseval=False):
        x.requires_grad_(True)
        if (iseval):
            torch.set_grad_enabled(True)
        output, dists, num_nn = self.forward(x, physical_particles)
        sdf = output[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        feature_vectors = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        if (iseval):
            torch.set_grad_enabled(False)
            sdf = sdf.detach()
            feature_vectors = feature_vectors.detach()
            gradients = gradients.detach()

        return sdf, feature_vectors, gradients, dists, num_nn

    def get_sdf_vals(self, x, physical_particles):
        sdf = self.forward(x, physical_particles)[0][:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf
    
    def search(self, ray_particles, particles, fix_radius):
        # particles (6320, 3) 不应该是(1, 6320, 3)
        # ray_particles (1024, 128, 3)
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

    def embedding_local_geometry(self, dists, indices, neighbors, radius, ray_particles):
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
        # smoothing 
        smoothed_pos, density = self.smoothing_position(ray_particles, neighbors, radius, num_nn, exclude_ray=self.encoding_cfg['exclude_ray'])
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
        return pos_like_feats, num_nn
    
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
