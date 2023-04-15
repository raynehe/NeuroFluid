"""
Load data exported from blender for the renderer
"""
import sys
sys.path.append('..')

import os
import json
import pickle as pkl
import joblib
import numpy as np
import os.path as osp
from PIL import Image
import cv2
from utils import rend_util

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from utils.ray_utils import get_ray_directions, get_rays

class BlenderDataset(Dataset):
    def __init__(self, root_dir, data_dir, data_names, cfg, imgW, imgH, start_index, end_index, imgscale, viewnames, split='train'):
        super(BlenderDataset, self).__init__()
        self.data_type = cfg.data_type
        # self.half_res = cfg.half_res
        self.viewnames = viewnames
        self.cfg = cfg
        self.start_index = start_index
        self.end_index = end_index
        self.split = split
        self.img_wh = (imgW, imgH)
        self.img_scale = imgscale
        assert self.img_wh[0] == self.img_wh[1], 'image width should be equal to image height'
        self.root_dir = root_dir #cfg.data_path
        self.data_dir = data_dir
        self.data_names = data_names #cfg.data_path
        self.transforms = T.ToTensor()
        # self.view_num = len(self.viewnames)
        self.read_metas(self.viewnames)
        self.read_box()
        print('Total dataset size:', self.all_rgbs_mv.shape[1])

    def read_metas(self, viewnames):
        self.all_rays_mv, self.all_rgbs_mv, self.all_cw_mv, self.focal_mv, self.particles_poss_mv, self.particles_vels_mv = [], [], [], [], [], []
        for iii, viewname in enumerate(viewnames):
            for data_name in self.data_names:
                _root_dir = osp.join(self.data_dir, data_name, viewname) # sim_003/view_15  sim_003/view_16
                all_rays_i, all_rgbs_i, all_cw_i, focal_i, particles_poss_i, particles_vels_i = self._read_meta(_root_dir)
                self.all_rays_mv.append(all_rays_i)
                self.all_rgbs_mv.append(all_rgbs_i)
                self.all_cw_mv.append(all_cw_i)
                self.focal_mv.append(focal_i)
            if iii == 0:
                self.particles_poss_mv.append(np.stack(particles_poss_i, 0))
                self.particles_vels_mv.append(np.stack(particles_vels_i, 0))
            print("iii",iii)
        self.all_rays_mv = np.stack(self.all_rays_mv, 0)
        self.all_rgbs_mv = np.stack(self.all_rgbs_mv, 0)
        self.all_cw_mv = np.stack(self.all_cw_mv, 0)
        # self.focal_mv = np.array(self.focal_mv)
        self.particles_poss_mv = np.stack(self.particles_poss_mv, 0)
        # (1, 50, 11532, 3)
        self.particles_vels_mv = np.stack(self.particles_vels_mv, 0)
        # import ipdb;ipdb.set_trace()
        
    def get_center_point(self, pose):
        # pose:4*4
        A = np.zeros((3, 4))
        b = np.zeros((3, 1))
        camera_centers=np.zeros((3, 1))

        P0 = pose[:3, :]

        K = cv2.decomposeProjectionMatrix(P0)[0]
        R = cv2.decomposeProjectionMatrix(P0)[1]
        c = cv2.decomposeProjectionMatrix(P0)[2]
        c = c / c[3]
        camera_centers[:,0]=c[:3].flatten()

        v = np.linalg.inv(K) @ np.array([800, 600, 1])
        v = v / np.linalg.norm(v)

        v=R[2,:]
        A[0:3, :3] = np.eye(3)
        A[0:3,3] = -v
        b[0:3] = c[:3]

        soll= np.linalg.pinv(A) @ b

        return soll,camera_centers

    def normalize_cameras(self, pose):
        # pose:4*4
        soll, camera_centers = self.get_center_point(pose)

        center = soll[:3].flatten()

        max_radius = np.linalg.norm((center[:, np.newaxis] - camera_centers), axis=0).max() * 1.1

        normalization = np.eye(4).astype(np.float32)

        normalization[0, 3] = center[0]
        normalization[1, 3] = center[1]
        normalization[2, 3] = center[2]

        normalization[0, 0] = max_radius / 3.0
        normalization[1, 1] = max_radius / 3.0
        normalization[2, 2] = max_radius / 3.0

        scale_mat = normalization.astype(np.float32)
        world_mat = pose.astype(np.float32)

        P = world_mat @ scale_mat
        P = P[:3, :4]
        _, pose = rend_util.load_K_Rt_from_P(None, P)
        return torch.from_numpy(pose).float()


    def _read_meta(self, root_dir):
        """
        read meta file. output rays and rgbs
        """
        with open(os.path.join(root_dir, f'transforms_{self.split}.json'), 'r') as f:
            self.meta = json.load(f)
    
        # parse
        # if self.half_res:
        #     W, H = self.img_wh[0] //2, self.img_wh[1] //2
        # else:
        #     W, H = self.img_wh
        W, H = int(self.img_wh[0] // self.img_scale), int(self.img_wh[1] // self.img_scale)
        focal = .5 * W / np.tan(0.5 * self.meta['camera_angle_x'])
        # get ray direction for all pixels
        directions = get_ray_directions(H, W, focal)
        image_paths = []
        poses = []
        all_rays = []
        all_rgbs = []
        all_cw = []
        # particles_path = []
        particle_poss = []
        particle_vels = []
        # self.all_mask = []
        # select_frame_idx = [_ for _ in range (self.start_index, self.end_index, 4)]
        # for idx in select_frame_idx:
        #     frame = self.meta['frames'][idx]
        idx = 0
        for frame in self.meta['frames'][self.start_index:self.end_index]:
            # get particles
            # particles_path.append(frame['particle_path'])
            if len(self.particles_poss_mv) == 0:
                particle_pos, particle_vel = self._read_particles(osp.join(root_dir, self.split, frame['particle_path']))
                particle_poss.append(particle_pos)
                particle_vels.append(particle_vel)
            # get orignal point and directrion
            pose = np.array(frame['transform_matrix'])[:3, :4]
            # pose = np.array(frame['transform_matrix'])
            # pose = self.normalize_cameras(pose)
            # pose = pose[:3, :4]
            poses.append(pose)
            print(np.linalg.det(pose[:3, :3]))
            c2w = torch.FloatTensor(pose)
            all_cw.append(pose)
            rays_o, rays_d = get_rays(directions, c2w)
            all_rays += [torch.cat([rays_o, rays_d], -1).numpy()]
            # read images
            image_path = osp.join(root_dir, '{}.png'.format(frame['file_path']))
            image_paths.append(image_path)
            image = Image.open(image_path)
            # if self.half_res:
            image = image.resize((int(self.img_wh[0]// self.img_scale), int(self.img_wh[1]// self.img_scale)), Image.ANTIALIAS)
            image = (np.asarray(image))/ 255.
            image = image.reshape(-1, 4)
            image = image[:, :3]*image[:, -1:] + (1-image[:, -1:])
            # image = self.transforms(image)
            # image = image.view(4, -1).permute(1,0) #(H*W, 4), RGBA image
            # image = image[:, :3]*image[:, -1:] + (1-image[:, -1:]) # blend A to RGB, assume white background. 
            all_rgbs.append(image)
            if not idx == 0:
                break
            idx += 1
        all_rays = np.stack(all_rays, 0)
        all_rgbs = np.stack(all_rgbs, 0)
        all_cw = np.stack(all_cw, 0)
        return all_rays, all_rgbs, all_cw, focal, particle_poss, particle_vels
        # return all_rays, all_rgbs, all_cw, focal, particles_path


    def read_box(self):
        bbox_path = self.meta['bounding_box']
        box_info = joblib.load(osp.join(self.root_dir, bbox_path))
        self.box = box_info['box']
        self.box_normals = box_info['box_normals']


    def _read_particles(self, particle_path):
        """
        read initial particle information and the bounding box information
        """
        if self.data_type == 'blender':
            # particle_info = np.load(osp.join(self.root_dir, self.split, particle_path))
            # with open(osp.join(self.root_dir, self.split, particle_path), 'rb') as fp:
            with open(particle_path, 'rb') as fp:
                particle_info = pkl.load(fp)
            particle_pos = np.array(particle_info['location']).reshape(-1, 3)
            particle_vel = np.array(particle_info['velocity']).reshape(-1, 3)
        elif self.data_type == 'splishsplash':
            # particle_info = np.load(osp.join(self.root_dir, self.split, particle_path))
            particle_info = np.load(particle_path)
            particle_pos = particle_info['pos']
            particle_vel = particle_info['vel']
        else:
            raise NotImplementedError('please enter correct data type')
        # import ipdb;ipdb.set_trace()
        # particle_pos = torch.from_numpy(particle_pos).float()
        # particle_vel = torch.from_numpy(particle_vel).float()
        return particle_pos, particle_vel


    def __getitem__(self, index):
        # rays = self.all_rays_mv[:, index]
        # rgbs = self.all_rgbs_mv[:, index]
        data = {}
        data['cw'] = torch.from_numpy(self.all_cw_mv[:,index]).float()
        data['rgb'] = torch.from_numpy(self.all_rgbs_mv[:, index]).float()
        data['rays'] = torch.from_numpy(self.all_rays_mv[:, index]).float()
        data['box'] = torch.from_numpy(self.box).float()
        data['box_normals'] = torch.from_numpy(self.box_normals).float()
        data['particles_pos'] = torch.from_numpy(self.particles_poss_mv[0, index]).float()
        data['particles_vel'] = torch.from_numpy(self.particles_vels_mv[0, index]).float()
        data['focal'] = self.focal_mv
        # data['view_name'] = self.viewnames
        # if index < self.all_rgbs_mv.shape[1]:
        data['cw_1'] = torch.from_numpy(self.all_cw_mv[:,index+1]).float()
        data['rays_1'] = torch.from_numpy(self.all_rays_mv[:, index+1]).float()
        data['rgb_1'] = torch.from_numpy(self.all_rgbs_mv[:, index+1]).float()
        data['particles_pos_1'] = torch.from_numpy(self.particles_poss_mv[0, index+1]).float()
        data['particles_vel_1'] = torch.from_numpy(self.particles_vels_mv[0, index+1]).float()
        return data

    def __len__(self):
        return self.all_rgbs_mv.shape[1]-1


if __name__ == '__main__':
    dataset = BlenderDataset()
    print('Done')