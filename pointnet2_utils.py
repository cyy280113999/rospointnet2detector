import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, n_group):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, n_group, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10 # init distance as infinity
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # init as random points
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(n_group):
        centroids[:, i] = farthest # first group is random , second is the fastest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask] # update distance, minimal distance from the group
        farthest = torch.max(distance, -1)[1] # farthest from the group
    return centroids


def query_ball_point(radius, group_size, xyz, centers):
    """
    Input:
        radius: local region radius
        group_size: max sample number in local region
        xyz: all points, [B, N, 3]
        centers: query points, [B, n_group, 3]
    Return:
        group_idx: grouped points index, [B, n_group, group_size]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, n_group, _ = centers.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, n_group, 1]) # init group as the all
    sqrdists = square_distance(centers, xyz)
    group_idx[sqrdists > radius ** 2] = N # set idx who is far from radius to nan 
    group_idx = group_idx.sort(dim=-1)[0][:, :, :group_size] # sort, N goes to end. cut to group_size
    group_first = group_idx[:, :, 0].view(B, n_group, 1).repeat([1, 1, group_size])
    mask = group_idx == N
    group_idx[mask] = group_first[mask] # if idx after cut includes N, replace by the first
    return group_idx # so make sure group is filled by point in ball. the lacked part is filled by first.


def sample_and_group(n_group, radius, group_size, xyz, features, returnfps=False):
    """
    Input:
        n_group: 
        radius:
        group_size: 
        xyz:        [B, N, 3]
        features:   [B, N, D]
    Return:
        grouped_xyz:        [B, n_group, group_size, 3]
        grouped_features:   [B, n_group, group_size, 3+D]
    """
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, n_group) # [B, n_group]
    group_centers = index_points(xyz, fps_idx) # center [B, n_group, C]
    grouped_idx = query_ball_point(radius, group_size, xyz, group_centers) # [B, n_group, group_size]
    grouped_xyz = index_points(xyz, grouped_idx) # [B, n_group, group_size, C]
    grouped_xyz_norm = grouped_xyz - group_centers.view(B, n_group, 1, C) # centering

    if features is not None: # select and add xyz to features
        grouped_features = index_points(features, grouped_idx)
        grouped_features = torch.cat([grouped_xyz_norm, grouped_features], dim=-1) # [B, n_group, group_size, C+D]
    else:
        grouped_features = grouped_xyz_norm
    # if returnfps:
    #     return group_centers, grouped_features, grouped_xyz, fps_idx
    return group_centers, grouped_features


def sample_and_group_all(xyz, features): # n_groups=1
    """
    Input:
        xyz:        [B, N, 3]
        features:   [B, N, D]
    Return:
        new_xyz:    [B, 1, 3]
        new_features: [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device) # only one point = 0
    grouped_xyz = xyz.view(B, 1, N, C)
    if features is not None:
        new_features = torch.cat([grouped_xyz, features.view(B, 1, N, -1)], dim=-1)
    else:
        new_features = grouped_xyz
    return new_xyz, new_features


class PointNetSetAbstraction(nn.Module):
    def __init__(self, n_groups, radius, group_size, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.n_groups = n_groups
        self.radius = radius
        self.group_size = group_size
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # NC 
        if features is not None:
            features = features.permute(0, 2, 1)

        if self.group_all:
            group_centers, grouped_features = sample_and_group_all(xyz, features)
        else:
            group_centers, grouped_features = sample_and_group(self.n_groups, self.radius, self.group_size, xyz, features)
        # group_centers: sampled    [B, n_groups, C]
        # grouped_features: sampled [B, n_groups, group_size, C+D]
        group_centers = group_centers.permute(0, 2, 1)
        grouped_features = grouped_features.permute(0, 3, 2, 1) # [B, C+D, group_size, n_groups]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_features =  F.relu(bn(conv(grouped_features)))
        grouped_features = torch.max(grouped_features, 2)[0] # [B, C+D, n_groups]
        return group_centers, grouped_features # group point, group features


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, n_groups, radius_list, group_size_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.n_groups = n_groups
        self.radius_list = radius_list
        self.group_size_list = group_size_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3 # feature+xyz
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if features is not None:
            features = features.permute(0, 2, 1)

        B, N, C = xyz.shape
        n_groups = self.n_groups
        group_centers = index_points(xyz, farthest_point_sample(xyz, n_groups)) # use same sampled centers
        grouped_features_list = []
        for i, radius in enumerate(self.radius_list):
            group_size = self.group_size_list[i]
            group_idx = query_ball_point(radius, group_size, xyz, group_centers)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= group_centers.view(B, n_groups, 1, C)
            if features is not None:
                grouped_features = index_points(features, group_idx)
                grouped_features = torch.cat([grouped_features, grouped_xyz], dim=-1)  # feature+xyz
            else:
                grouped_features = grouped_xyz
            grouped_features = grouped_features.permute(0, 3, 2, 1)  # [B, D, group_size, n_groups]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_features =  F.relu(bn(conv(grouped_features)))
            grouped_features = torch.max(grouped_features, 2)[0]  # [B, D', n_groups]
            grouped_features_list.append(grouped_features)
        group_centers = group_centers.permute(0, 2, 1)
        grouped_features = torch.cat(grouped_features_list, dim=1)
        return group_centers, grouped_features


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

