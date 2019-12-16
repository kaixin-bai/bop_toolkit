import numpy as np
from numpy import ma
import open3d as o3d
import os
from os import listdir
from PIL import Image
from matplotlib import pyplot as plt
import json
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import csv
from lib.network_6d import PoseNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.transform_tool import *
from visualizations.vis_6d_pose_estimation import vis_6d_pose_estimation
import time


class PoseWriter: # load data
    def __init__(self):
        self.dataset_path = "/data/hdd1/kb/agile/bkx_master/6dofbkx/datasets/ycbv/ycbv/test"    # should contain the dataset folders
        self.scene_id_list = listdir(self.dataset_path)
        self.pe = PoseEstimate()
    def get_R_and_t(self, T):
        R = np.array([T[0,0],T[0,1],T[0,2],T[1,0],T[1,1],T[1,2],T[2,0],T[2,1],T[2,2]])
        t = np.array([T[0,3],T[1,3],T[2,3]])
        return R, t
    def __getitem__(self, item):    # item is for scene_id_list
        scene_id_str = self.scene_id_list[item]
        im_id_str_list = [f[:-4] for f in listdir(str(self.dataset_path+'/'+scene_id_str+'/depth/'))]
        # got scene_gt_json as directory, key is im_scene
        scene_camera_json = json.load(open(self.dataset_path + '/' + scene_id_str + '/scene_camera.json'))
        scene_gt_json = json.load(open(self.dataset_path + '/' + scene_id_str + '/scene_gt.json'))
        scene_gt_info_json = json.load(open(self.dataset_path + '/' + scene_id_str + '/scene_gt_info.json'))

        with open('/data/hdd1/kb/agile/bkx_master/6dofbkx/datasets/ycbv/kx-iros15_ycbv-test.csv', 'a') as csvfile:  # define the csv first
            writer = csv.writer(csvfile)
            for im_scene_int, obj_dics in scene_gt_json.items():
                im_scene_int = int(im_scene_int)
                # print('im scene int:',im_scene_int)
                # print('obj_list:', obj_dics[0])
                for obj_index in range(len(obj_dics)):
                    start = time.time()
                    obj_id_int = obj_dics[obj_index]['obj_id']
                    # print('obj id:', obj_id_int)
                    # print('obj index', obj_index)
                    rgb = np.asarray(Image.open(self.dataset_path+'/'+scene_id_str+'/rgb/'+"%06d"%(im_scene_int)+".png"))
                    depth = np.asarray(Image.open(self.dataset_path+'/'+scene_id_str+'/depth/'+"%06d"%(im_scene_int)+".png"))
                    mask = np.asarray(Image.open(
                        self.dataset_path + '/' + scene_id_str + '/mask/' + "%06d" % (im_scene_int) + '_' + "%06d" % (
                            obj_index) + ".png"))
                    # if 0:
                    #     plt.figure()
                    #     plt.subplot(1,3,1)
                    #     plt.imshow(rgb)
                    #     plt.subplot(1, 3, 2)
                    #     plt.imshow(depth)
                    #     plt.subplot(1, 3, 3)
                    #     plt.imshow(mask)
                    #     plt.show()

                    # self.pe.get_pcd(rgb,depth)
                    # self.pe.get_masked_pcd(rgb, depth, mask)
                    transformation = self.pe.pose_estimate(rgb,depth,mask, obj_id_int-1)
                    # print(transformation)
                    # ["scene_id","im_id","obj_id","score","R","t","time"]
                    R, t = self.get_R_and_t(transformation)
                    t = t * 1000.0
                    # TODO: not finish yet
                    # print([int(scene_id_str), int(im_scene_int),"", list(R), list(t), float(time.time()-start)])
                    R_str = "{} {} {} {} {} {} {} {} {}".format(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7] , R[8])
                    # print(R_str)
                    r_str = "{} {} {}".format(t[0], t[1], t[2])
                    writer.writerow([int(scene_id_str), int(im_scene_int), obj_id_int, "",str(R_str), str(r_str), float(time.time()-start)])

                    # writer.writerow([int(scene_id_str), int(im_scene_int), obj_id_int, "",str(R)[1:-1], str(t)[1:-1], float(time.time()-start)])

class PoseEstimate:
    def __init__(self):
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic.intrinsic_matrix = np.array([[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]])
        self.depth_scale = 10000.0
        # --------------------------------------------------------------------------------------------------------------
        self.model_path = '/data/hdd1/kb/agile/bkx_master/6dofbkx/trained_models/ycb/v_6d/pose_model_59_0.03942583555500309.pth'
        self.num_points = 1000
        self.num_obj = 21
        # read object point cloud model
        # --------------------------------------------------------------------------------------------------------------
        class_file = open('/data/hdd1/kb/agile/bkx_master/6dofbkx/datasets/ycb/dataset_config/classes.txt')
        class_id = 0
        cld = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            class_input = class_input[:-1]
            obj_pcds_filepath = '{0}/models/{1}/points.xyz'.format('/data/hdd1/kb/agile/densefusion/src/DenseFusion/datasets/ycb/YCB_Video_Dataset',
                                                                   class_input)
            input_file = open(obj_pcds_filepath)
            cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1]
                input_line = input_line.split(' ')
                cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            input_file.close()
            cld[class_id] = np.array(cld[class_id])
            class_id += 1
        self.cld = cld
        # --------------------------------------------------------------------------------------------------------------
        estimator = PoseNet(num_points=self.num_points, num_obj=self.num_obj)
        estimator.cuda()
        estimator = nn.DataParallel(estimator, device_ids=[0])
        estimator.load_state_dict(torch.load(self.model_path))
        estimator.eval()
        self.estimator = estimator
        # --------------------------------------------------------------------------------------------------------------

    def get_pcd(self, color, depth, vis=False):
        img = np.array(color)
        depth = np.array(depth.reshape(depth.shape[0],depth.shape[1],1)).astype(np.float32) / self.depth_scale # to meter
        detph = o3d.geometry.Image(depth)
        img = o3d.geometry.Image(img)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, detph, depth_scale=1.0,
                                                                  convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        if vis:
            o3d.visualization.draw_geometries([pcd])
        return pcd

    def get_masked_pcd(self, color, depth, mask, vis=False):
        depth = depth.reshape(depth.shape[0],depth.shape[1],1)
        mask = ma.masked_equal(mask, 255)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        intrinsic = self.intrinsic
        color = color * mask
        depth = depth * mask
        img = np.array(color)
        depth = np.array(depth).astype(np.float32) / self.depth_scale
        detph = o3d.geometry.Image(depth)
        img = o3d.geometry.Image(img)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, detph, depth_scale=1.0,
                                                                  convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        if vis:
            o3d.visualization.draw_geometries([pcd])
        return pcd

    def pose_estimate(self, color, depth, mask, idx, vis=False):
        pcd = self.get_masked_pcd(color, depth, mask)
        # --------------- generate input data --------------------
        cloudxyz = np.asarray(pcd.points)
        cloudrgb = np.asarray(pcd.colors)
        cloudxyzrgb = np.concatenate((cloudxyz, cloudrgb), axis=1)
        np.random.shuffle(cloudxyzrgb)
        cloudxyzrgb = cloudxyzrgb[:self.num_points,:6]
        cloud = cloudxyzrgb[:, :3]
        cloudrgb = cloudxyzrgb[:,3:6] * 255
        cloudxyzrgb = np.concatenate((cloud, cloudrgb), axis=1)
        cloudxyzrgb = torch.from_numpy(cloudxyzrgb.astype(np.float32))
        cloudxyzrgb = cloudxyzrgb.view(1, self.num_points, 6)
        obj_idx = torch.LongTensor([idx])
        obj_idx = Variable(obj_idx).cuda()
        obj_idx = obj_idx.view(1, 1)
        cloud = torch.from_numpy(cloud.astype(np.float32))
        cloud = Variable(cloud).cuda()
        cloud = cloud.view(1, self.num_points, 3)
        pred_r_6d, pred_t, pred_c = self.estimator(cloudxyzrgb, obj_idx)
        pred_r_matrix = compute_rotation_matrix_from_ortho6d(pred_r_6d[0]).view(1, self.num_points, 9)
        pred_c = pred_c.view(1, self.num_points)  # bs=1
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(self.num_points, 1, 3)
        points = cloud.view(self.num_points, 1, 3)
        my_r = pred_r_matrix[0][which_max[0]].view(-1).cpu().data.numpy()  # 9个值，是旋转矩阵
        my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        T = np.array([[my_r[0],my_r[1],my_r[2],my_t[0]],[my_r[3],my_r[4],my_r[5],my_t[1]],[my_r[6],my_r[7],my_r[8],my_t[2]],[0,0,0,1]])
        # load obj point cloud model
        objmodel = o3d.geometry.PointCloud()
        objmodel.points = o3d.utility.Vector3dVector(self.cld[idx])
        if vis:
            o3d.visualization.draw_geometries([self.get_pcd(color, depth), deepcopy(objmodel).transform(T)])
        # icp part
        # -------------------------------------fine alignment-----------------------------------------------------------
        new_pcd_source = deepcopy(objmodel)
        pcd_target = pcd
        new_pcd_source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.0001, max_nn=30))
        pcd_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.0001, max_nn=30))
        current_transformation = T
        result_icp = o3d.registration.registration_icp(
            new_pcd_source, pcd_target, 0.02, current_transformation,
            o3d.registration.TransformationEstimationPointToPoint())
        # print('total time(include ICP):', time() - start)
        current_transformation = result_icp.transformation
        flip_rot_matrix = np.array(
            [[1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32)
        # o3d.visualization.draw_geometries([deepcopy(self.get_pcd(color, depth)).transform(flip_rot_matrix), deepcopy(objmodel).transform(current_transformation).transform(flip_rot_matrix)])
        return current_transformation

def main():
    # eval_dataset_path = "/data/hdd1/kb/agile/bkx_master/6dofbkx/datasets/ycb-v"
    pw = PoseWriter()
    # pw[0]
    for i in range(12):
        pw[i]


if __name__ == '__main__':
    main()
