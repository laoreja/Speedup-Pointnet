""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2016
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import group_point, knn_point, query_ball_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import tf_util

#def gather_point(features, idx):
#    '''
#    input:
#        batch_size * x * ndataset   float32
#        batch_size * npoints        int32
#    returns:
#        batch_size * x * npoints    float32
#    '''
#    features_shape = features.get_shape()
#    batch_size = features_shape[0].value
#    ndatasets = features_shape[2].value
#    nfeatures = features_shape[1].value
#    npoint = idx.get_shape()[1].value
#    
#    reshaped_features = tf.reshape(tf.transpose(features, [0,2,1]), [-1, nfeatures])
#    offsets = tf.tile(tf.range(batch_size)[:, None], [1, npoint]) * ndatasets + idx
#    new_features = tf.reshape(tf.gather(reshaped_features, offsets), [batch_size, npoint, -1])
#    new_features = tf.transpose(new_features, [0,2,1])
#        
#    return new_features

def sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec=None, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        tnet_spec: dict (keys: mlp, mlp2, is_training, bn_decay), if None do not apply tnet
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''
    # new_xyz: Seems like the xyz coords of the centroids
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)

    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx,_ = query_ball_point(radius, nsample, xyz, new_xyz)
#        idx,_ = query_square_point_wrapper(radius * 2 / 3, nsample, xyz, new_xyz)#query_ball_point(radius, nsample, xyz, new_xyz)
        
    # group all points according to idx, xyz (translated) & points' features
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.expand_dims(new_xyz, 2) # translation normalization
    if tnet_spec is not None:
        grouped_xyz = tnet(grouped_xyz, tnet_spec)
    if points is not None:
        new_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, new_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.zeros([batch_size,1,3], dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.tile(tf.reshape(tf.range(nsample), (1,1,nsample)), (batch_size,1,1)) # (batch_size, 1, nsample)
    grouped_xyz = tf.expand_dims(xyz, 1) #tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    # Q: xyz: (batch_size, ndataset, 3) -> (batch_size, npoint=1, nsample, 3)?
    # A: in group all, nsample == ndataset
    if points is not None:
        if use_xyz:
            points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        points = tf.expand_dims(points, 1) # (batch_size, 1, 16, 259)
    else:
        points = grouped_xyz
    return new_xyz, points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', tnet_spec=None, knn=False, use_xyz=True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    print('in pointnet_sa_module')
    print('points: ', points)
    print('xyz: ', xyz)
    with tf.variable_scope(scope) as sc:
        if points == None:
            all_points = xyz
        else:
            if use_xyz:
                all_points = tf.concat([xyz, points], 1)
        all_points = all_points[:, :, :, None]
        all_points, old_conv, kernel = tf_util.conv2d_return_conv(all_points, 
                                    mlp[0], 
                                    [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=bn, is_training=is_training,
                                    scope='all_conv%d'%(0), bn_decay=bn_decay) 
        # (batch_size, npoint, nsample, last_mlp)
        
        xyz = tf.transpose(xyz, [0,2,1])
        if points is not None:
            points = tf.transpose(points, [0,2,1])        
        
        if group_all:
            xyz, points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            xyz, points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec, knn, use_xyz)

        xyz = tf.transpose(xyz, [0,2,1])
        points = tf.transpose(points, [0,3,1,2])
        
        # points: (batch_size, npoint, nsample, 3+channel)
        print 'old_conv', old_conv
        old_conv = tf.transpose(group_point(tf.transpose(old_conv, [0,2,1]), idx), [0,3,1,2])
        print 'old_conv', old_conv
        for i, num_out_channel in enumerate(mlp):
            if i == 0:
                do_add = True
                t = -1*xyz
            else:
                do_add = False
                old_conv = None
                t = None
                kernel = None
            points = tf_util.conv2d(points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        do_add=do_add, old_conv=old_conv, t=t, 
                                        kernel=kernel) 
            # (batch_size, npoint, nsample, last_mlp)

        points = tf.reduce_max(points, axis=[3], keep_dims=False)
        # (batch_size, npoint, 1, num_features)
        return xyz, points, idx

