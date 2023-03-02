"""
Renderer
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from .nerf import Embedding, NeRF
from utils.ray_utils import coarse_sample_ray
from utils.ray_utils import ImportanceSampling

from pytorch3d.ops import ball_query

from models.embedder import *
from models.implicit_network import ImplicitNetwork
from models.rendering_network import RenderingNetwork
from models.density import LaplaceDensity
from models.ray_sampler import ErrorBoundSampler
class RenderNet(nn.Module):
    def __init__(self, cfg, near, far):
        super(RenderNet, self).__init__()
        self.cfg = cfg
        # self.ray_chunk_interpolate = self.cfg.ray_chunk_interpolate
        # self.ray_chunk = cfg.ray_chunk
        self.near = near
        self.far = far
        self.N_samples = cfg.ray.N_samples
        self.N_importance = cfg.ray.N_importance
        self.raduis = self.cfg.NN_search.search_raduis_scale * self.cfg.NN_search.particle_radius
        self.fix_radius = self.cfg.NN_search.fix_radius
        self.num_neighbor = self.cfg.NN_search.N_neighbor

        # build network
        self.embedding_xyz = Embedding(3, 10)
        in_channels_xyz = self.embedding_xyz.out_channels
        self.embedding_dir = Embedding(3, 4)
        in_channels_dir = self.embedding_dir.out_channels
        if cfg.encoding.density:
            self.embedding_density = Embedding(1, 4)
            in_channels_xyz += self.embedding_density.out_channels
        if cfg.encoding.var:
            in_channels_xyz += self.embedding_xyz.out_channels
        if cfg.encoding.smoothed_pos:
            in_channels_xyz += self.embedding_xyz.out_channels
        if cfg.encoding.smoothed_dir:
            in_channels_dir += self.embedding_dir.out_channels
        # self.nerf_coarse = NeRF(in_channels_xyz=in_channels_xyz, in_channels_dir=in_channels_dir)
        # self.nerf_fine = NeRF(in_channels_xyz=in_channels_xyz, in_channels_dir=in_channels_dir)

        # conf_filepath = 'configs/nf2model.conf'
        # # conf_filepath = 'NeuroFluid/configs/nf2model.conf' # for testing
        # conf = ConfigFactory.parse_file(conf_filepath).get_config('model')
        
        # self.feature_vector_size = conf.get_int('feature_vector_size')
        # self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        # self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        # self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()
        # self.density = LaplaceDensity(**conf.get_config('density'))
        # self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        # self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        # self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.feature_vector_size = int(cfg.feature_vector_size)
        self.scene_bounding_sphere = float(cfg.scene_bounding_sphere)
        self.white_bkgd = bool(cfg.white_bkgd)
        self.bg_color = torch.tensor(list(cfg.bg_color)).float().cuda()
        self.density = LaplaceDensity(**cfg.density)
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **cfg.ray_sampler)
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **cfg.implicit_network)
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **cfg.rendering_network)

    def set_ro(self, cw):
        """
        get the camera position in world coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
        """
        ray_o = cw[:,3] # (3,)
        return ray_o

    
    def get_particles_direction(self, particles, ro):
        ros = ro.expand(particles.shape[0], -1)
        dirs = particles - ros
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        return dirs


    def mask_gather(self, points, idx):
        N, P, D = points.shape
        K = idx.shape[2]
        # Match dimensions for points and indices
        idx_expanded = idx[..., None].expand(-1, -1, -1, D)
        points = points[:, :, None, :].expand(-1, -1, K, -1)
        idx_expanded_mask = idx_expanded.eq(-1)
        idx_expanded = idx_expanded.clone()
        # Replace -1 values with 0 for gather
        idx_expanded[idx_expanded_mask] = 0
        # Gather points
        selected_points = points.gather(dim=1, index=idx_expanded)
        # Replace padded values
        selected_points[idx_expanded_mask] = 0.0
        return selected_points


    def _ball_query(self, query_points, points, radius, nsample):
        dists = torch.cdist(query_points, points) # N_ray, N_ray_points, N_phys_points
        dists_sorted, indices_sorted = torch.sort(dists, dim=-1)
        _dists, _indices = dists_sorted[:, :, :nsample], indices_sorted[:, :, :nsample]
        mask = (_dists > radius)
        _dists[mask] = 0.
        _indices[mask] = -1
        selected_points = self.mask_gather(points, _indices)
        return _dists, _indices, selected_points


    def get_search_raduis(self, R, z, f):
        dist = R * torch.abs(z) / f
        return dist.unsqueeze(-1)
    
    
    def smoothing_position(self, ray_pos, nn_poses, raduis, num_nn, exclude_ray=True, larger_alpha=0.9, smaller_alpha=0.1):
        dists = torch.norm(nn_poses - ray_pos.unsqueeze(-2), dim=-1)
        weights = torch.clamp(1 - (dists / raduis) ** 3, min=0)
        weighted_nn = (weights.unsqueeze(-1) * nn_poses).sum(-2) / (weights.sum(-1, keepdim=True)+1e-12)
        if exclude_ray:
            pos = weighted_nn
        else:
            if self.cfg.encoding.same_smooth_factor:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
            else:
                alpha = torch.ones(ray_pos.shape[0], ray_pos.shape[1], 1) * larger_alpha
                alpha[num_nn.le(20)] = smaller_alpha
            pos = ray_pos * (1-alpha) + weighted_nn * alpha
        return pos, weights.sum(-1, keepdim=True)
    
    
    def search(self, ray_particles, particles, fix_radius):
        raw_data = particles.unsqueeze(0).repeat(ray_particles.shape[0], 1, 1)
        # plot
        # self.plot_3d( ray_particles.cpu().numpy(), raw_data.cpu().numpy())
        # ray_particles [512,128,3]: giving a batch of 512 point clouds
        # raw_data [512, 10212, 3] : giving a batch of 512 point clouds
        # dists [512, 128, 20]     : Tensor of shape (N, P1, K) 
        if fix_radius:
            radiis = self.raduis
            dists, indices, neighbors = ball_query(p1=ray_particles, 
                                                   p2=raw_data, 
                                                   radius=radiis, K=self.num_neighbor)
        # else:
        #     radiis = self.get_search_raduis(self.raduis, ray_particles[:,:,-1] - ro[-1], focal)
        #     dists, indices, neighbors = self._ball_query(ray_particles, raw_data, radiis, self.num_neighbor)
        return dists, indices, neighbors, radiis
    
        
    def embedding_local_geometry(self, dists, indices, neighbors, radius, ray_particles, rays, ro, sigma_only=False):
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
        hit_pos = ray_particles.reshape(-1,3) # [65536, 3]
        hit_pos_embedded = self.embedding_xyz(hit_pos) # [65536, 21*3]
        pos_like_feats.append(hit_pos_embedded)
        if not sigma_only:
            hit_dir = rays[:,3:]
            hit_dir_embedded = self.embedding_dir(hit_dir)
            hit_dir_embedded = torch.repeat_interleave(hit_dir_embedded, repeats=ray_particles.shape[1], dim=0)
            dir_like_feats = []
            dir_like_feats.append(hit_dir_embedded)
        # smoothing 
        smoothed_pos, density = self.smoothing_position(ray_particles, neighbors, radius, num_nn, exclude_ray=self.cfg.encoding.exclude_ray)
        smoothed_dir = self.get_particles_direction(smoothed_pos.reshape(-1, 3), ro)
        # density
        if self.cfg.encoding.density:
            density_embedded = self.embedding_density(density.reshape(-1, 1))
            pos_like_feats.append(density_embedded)
        # smoothed pos
        if self.cfg.encoding.smoothed_pos:
            smoothed_pos_embedded = self.embedding_xyz(smoothed_pos.reshape(-1, 3))
            pos_like_feats.append(smoothed_pos_embedded)
        # variance
        if self.cfg.encoding.var:
            vec_pp2rp = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, 3).to(neighbors.device)
            vec_pp2rp[nn_mask] = (neighbors - ray_particles.unsqueeze(-2))[nn_mask]
            vec_pp2rp_mean = vec_pp2rp.sum(-2) / (num_nn+1e-12)
            variance = torch.zeros(ray_particles.shape[0], ray_particles.shape[1], self.num_neighbor, 3).to(neighbors.device)
            variance[nn_mask] = ((vec_pp2rp - vec_pp2rp_mean.unsqueeze(-2))**2)[nn_mask]
            variance = variance.sum(-2) / (num_nn+1e-12)
            variance_embedded = self.embedding_xyz(variance.reshape(-1,3))
            pos_like_feats.append(variance_embedded)
        # smoothed dir
        if self.cfg.encoding.smoothed_dir:
            smoothed_dir_embedded = self.embedding_dir(smoothed_dir)
            dir_like_feats.append(smoothed_dir_embedded)
        if not sigma_only:
            return pos_like_feats, dir_like_feats, num_nn
        else:
            return pos_like_feats
        
    
    def render_image(self, rgbsigma, zvals, rays, noise_std, white_background, sdf):
        rgbs = rgbsigma[..., :3]
        # sigmas = rgbsigma[..., 3]
        sigmas_flat = self.density(sdf) # density
        sigmas = sigmas_flat.reshape(-1, zvals.shape[1])
        # convert these values using volume rendering (Section 4)
        deltas = zvals[:, 1:] - zvals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
        
        deltas = deltas * torch.norm(rays[:,3:].unsqueeze(1), dim=-1)
        
        noise = 0.
        if noise_std > 0.:
            noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
        
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        weights_sum = weights.sum(1)
        
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
        depth_final = torch.sum(weights*zvals, -1) # (N_rays)
        
        if white_background:
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)
        return rgb_final, depth_final, weights
    
    def plot_3d(self, particles, querypoints):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(particles[:,0], particles[:,1], particles[:,2], s=1, c='r', marker='.', alpha=0.1)
        # plt.show()
        ax.scatter(querypoints[:,0], querypoints[:,1], querypoints[:,2], s=1, c='b', marker='.', alpha=0.1)
        ax.legend()
        plt.show()
        plt.close()
    
    def plot_sdf(self,particles, sdf):
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(particles[:,0], particles[:,1], particles[:,2], s=sdf, c='r', marker='.', alpha=0.1)

        x, y, z = particles[:,0], particles[:,1], particles[:,2]
        v = sdf[:]
        min_v = min(v)
        max_v = max(v)
        color = [plt.get_cmap("seismic", 100)(int(float(i-min_v)/(max_v-min_v)*100)) for i in v]

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        # 设置colormap，与上面提到的类似，使用"seismic"类型的colormap，共100个级别
        plt.set_cmap(plt.get_cmap("seismic", 100))
        # 绘制三维散点，各个点颜色使用color列表中的值，形状为"."
        im = ax.scatter(x, y, z, s=100,c=color,marker='.')
        # 2.1 增加侧边colorbar
        # 设置侧边colorbar，colorbar上显示的值使用lambda方程设置
        fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x,pos:int(x*(max_v-min_v)+min_v)))
        # 2.2 增加坐标轴标签
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # 2.3显示
        plt.show()



        # fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x,pos:int(x*(max_v-min_v)+min_v)))
        # ax.legend()
        # plt.show()
        # plt.close()

    def forward(self, physical_particles, ro, rays, focal, c2w, iseval = False, use_disp=False, perturb=0, noise_std=0., white_background=True):
        """
        physical_particles: N_particles, 3
        ray_particles: N_ray, N_samples, 3
        zvals: N_rays, N_samples
        ro: 3, camera location
        rays: N_rays, 6
        """
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # (1024, 3) ray_o=ro
        num_pixels = self.N_samples
        results = {}
        N_rays = rays.shape[0]
        # ---------------
        # coarse render
        # ---------------
        # coarsely sample
        z_values_0, z_samples_eik = self.ray_sampler.get_z_vals(rays_d, rays_o, physical_particles, rays, ro, self) # [1024,98]  [1024,1]
        ray_particles_0 = rays_o.unsqueeze(1) + z_values_0.unsqueeze(2) * rays_d.unsqueeze(1)
        N_samples = z_values_0.shape[1]
        self.N_samples = N_samples
        # search
        # ray_particles_0 [1024,98,3]
        # physical_particles [10212, 3]
        # dists_0 [1024,98,20]
        # indices_0 [1024,98,20]
        # neighbors_0 [1024,98,20,3]
        # plot
        ray_particles_0_plain = ray_particles_0.reshape(-1, 3)
        # self.plot_3d( physical_particles.cpu().numpy(), ray_particles_0_plain.cpu().numpy())
        dists_0, indices_0, neighbors_0, radius_0 = self.search(ray_particles_0, physical_particles, self.fix_radius)
        # embedding attributes
        pos_like_feats_0, dirs_like_feats_0, num_nn_0 = self.embedding_local_geometry(dists_0, indices_0, neighbors_0, radius_0, ray_particles_0, rays, ro)
        input_feats_0 = torch.cat(pos_like_feats_0+dirs_like_feats_0, dim=1)
        pos_input_feats_0 = torch.cat(pos_like_feats_0, dim=1)
        dir_input_feats_0 = torch.cat(dirs_like_feats_0, dim=1)
        # predict rgbsigma
        # rgbsigma_0 = self.nerf_coarse(input_feats_0)
        # mask_0 [512, 98, 1]
        mask_0 = torch.all(dists_0!=0, dim=-1, keepdim=True).float()
        # if self.cfg.use_mask:
        #     rgbsigma_0 = rgbsigma_0.view(-1, self.N_samples, 4)*mask_0
        # else:
        #     rgbsigma_0 = rgbsigma_0.view(-1, self.N_samples, 4)
        # rgbs = rgbsigma_0[..., :3]
        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(pos_input_feats_0, iseval = iseval)
        self.plot_sdf(ray_particles_0_plain.cpu().numpy(), sdf.cpu().numpy())
        rgb_flat = self.rendering_network(pos_input_feats_0, gradients, dir_input_feats_0, feature_vectors)
        if self.cfg.use_mask:
            rgbs = rgb_flat.view(-1, self.N_samples, 3)*mask_0
        else:
            rgbs = rgb_flat.view(-1, self.N_samples, 3)
        # rgbsigma_0[..., :3] = rgbs[:] # 现在rgbsigma_0是volsdf得到的rgb + nf得到的sigma
        # render
        rgb_final_0, depth_final_0, weights_0 = self.render_image(rgbs, z_values_0, rays, noise_std, white_background, sdf)
        # rgb_final_0, depth_final_0, weights_0 = self.volume_rendering(rgbs, z_values_0, white_background, sdf)
        if self.training:
            results['grad_theta'] = self.eikonal_loss(num_pixels, rays_o, rays_d, z_samples_eik, physical_particles, ro, rays)
        if not self.training:
            gradients = gradients.detach()
        results['rgb0'] = rgb_final_0
        results['depth0'] = depth_final_0
        results['opacity0'] = weights_0.sum(1)
        results['num_nn_0'] = num_nn_0
        results['mask_0'] = mask_0.sum(1)
        return results


    def coarse_rendering(self, physical_particles, ro, rays, focal, c2w, use_disp=False, perturb=0, noise_std=0., white_background=True):
        """
        physical_particles: N_particles, 3
        ray_particles: N_ray, N_samples, 3
        zvals: N_rays, N_samples
        ro: 3, camera location
        rays: N_rays, 6
        """
        results = {}
        N_rays = rays.shape[0]
        # ---------------
        # coarse render
        # ---------------
        # coarsely sample
        z_values_0, ray_particles_0 = coarse_sample_ray(self.near, self.far, rays, self.N_samples, use_disp, perturb)
        # search
        dists_0, indices_0, neighbors_0, radius_0 = self.search(ray_particles_0, physical_particles, self.fix_radius)
        # embedding attributes
        pos_like_feats_0, dirs_like_feats_0, num_nn_0 = self.embedding_local_geometry(dists_0, indices_0, neighbors_0, radius_0, ray_particles_0, rays, ro)
        input_feats_0 = torch.cat(pos_like_feats_0+dirs_like_feats_0, dim=1)
        # predict rgbsigma
        rgbsigma_0 = self.nerf_coarse(input_feats_0)
        mask_0 = torch.all(dists_0!=0, dim=-1, keepdim=True).float()
        if self.cfg.use_mask:
            rgbsigma_0 = rgbsigma_0.view(-1, self.N_samples, 4)*mask_0
        else:
            rgbsigma_0 = rgbsigma_0.view(-1, self.N_samples, 4)
        # render
        rgb_final_0, depth_final_0, weights_0 = self.render_image(rgbsigma_0, z_values_0, rays, noise_std, white_background)
        results['rgb0'] = rgb_final_0
        results['depth0'] = depth_final_0
        results['opacity0'] = weights_0.sum(1)
        results['num_nn_0'] = num_nn_0
        results['mask_0'] = mask_0.sum(1)
        return results

    
    def fine_rendering(self, physical_particles, ro, rays, focal, c2w, use_disp=False, perturb=0, noise_std=0., white_background=True):
        results = {}
        # ---------------
        # coarse render
        # ---------------
        # coarsely sample
        z_values_0, ray_particles_0 = coarse_sample_ray(self.near, self.far, rays, self.N_samples, use_disp, perturb)
        # search
        dists_0, indices_0, neighbors_0, radius_0 = self.search(ray_particles_0, physical_particles, self.fix_radius)

        # --------
        # only need sigma
        pos_like_feats,= self.embedding_local_geometry(dists_0, indices_0, neighbors_0, radius_0, ray_particles_0, rays, ro, sigma_only=True)
        input_feats = torch.cat(pos_like_feats, dim=1)
        sigmas_0 = self.nerf_coarse(input_feats, sigma_only=True)
        mask_0 = torch.all(dists_0!=0, dim=-1, keepdim=True).float()
        if self.cfg.use_mask:
            sigmas_0 = (sigmas_0.view(-1, self.N_samples, 1)*mask_0).squeeze(-1)
        else:
            sigmas_0 = (sigmas_0.view(-1, self.N_samples, 1)).squeeze(-1)
        # convert these values using volume rendering (Section 4)
        deltas = z_values_0[:, 1:] - z_values_0[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
        deltas = deltas * torch.norm(rays[:,3:].unsqueeze(1), dim=-1)
        noise = 0.
        if noise_std > 0.:
            noise = torch.randn(sigmas_0.shape, device=sigmas_0.device) * noise_std
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas_0+noise)) # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
        weights_0 = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1] # (N_rays, N_samples_)
        
        # ---------------
        # fine render
        # ---------------
        assert self.N_importance>0
        if True:
            ray_particles_1, z_values_1 = ImportanceSampling(z_values_0, weights_0, self.N_importance, rays[...,:3], rays[...,3:], det=(perturb==0))
            # search
            dists_1, indices_1, neighbors_1, radius_1 = self.search(ray_particles_1, physical_particles, self.fix_radius)
            # embedding attributes
            pos_like_feats_1, dirs_like_feats_1, num_nn_1 = self.embedding_local_geometry(dists_1, indices_1, neighbors_1, radius_1, ray_particles_1, rays, ro)
            input_feats_1 = torch.cat(pos_like_feats_1+dirs_like_feats_1, dim=1)
            # predict rgbsigma
            rgbsigma_1 = self.nerf_fine(input_feats_1)
            mask_1 = torch.all(dists_1!=0, dim=-1, keepdim=True).float()
            if self.cfg.use_mask:
                rgbsigma_1 = rgbsigma_1.view(-1, mask_1.shape[1], 4)*mask_1
            else:
                rgbsigma_1 = rgbsigma_1.view(-1, mask_1.shape[1], 4)
            # render
            rgb_final_1, depth_final_1, weights_1 = self.render_image(rgbsigma_1, z_values_1, rays, noise_std, white_background)
            results['rgb1'] = rgb_final_1
            results['depth1'] = depth_final_1
            results['opacity1'] = weights_1.sum(1)
            results['num_nn_1'] = num_nn_1
            results['mask_1'] = mask_1.sum(1)
        return results
        
    def volume_rendering(self, rgbsigma, z_vals, white_background, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])
        # density = rgbsigma[..., 3]

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here
        rgbs = rgbsigma[..., :3]
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, 1)
        depth_final = torch.sum(weights*z_vals, -1)
        if white_background:
            weights_sum = weights.sum(1)
            rgb_final = rgb_final + 1-weights_sum.unsqueeze(-1)
        return rgb_final, depth_final, weights
        
    def eikonal_loss(self, num_pixels, cam_loc, ray_dirs, z_samples_eik, physical_particles, ro, rays):
        # Sample points for the eikonal loss
        batch_size = rays.shape[0]
        n_eik_points = batch_size * num_pixels
        eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()
        # eikonal_points [65536, 3]

        # add some of the near surface points
        eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
        eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
        # eik_near_points [1024, 3]
        # eikonal_points [66560, 3]

        eikonal_points = eikonal_points.reshape(eik_near_points.shape[0], eikonal_points.shape[0]//eik_near_points.shape[0], eikonal_points.shape[1])
        # eikonal_points [1024,65,3]
        dists, indices, neighbors, radius = self.search(eikonal_points, physical_particles, self.fix_radius)
        eikonal_pos_like_feats, _, _ = self.embedding_local_geometry(dists, indices, neighbors, radius, eikonal_points, rays, ro)
        eikonal_pos_like_feats = torch.cat(eikonal_pos_like_feats, dim=1)
        grad_theta = self.implicit_network.gradient(eikonal_pos_like_feats)
        
        # grad_theta = self.implicit_network.gradient(eikonal_points)
        return grad_theta