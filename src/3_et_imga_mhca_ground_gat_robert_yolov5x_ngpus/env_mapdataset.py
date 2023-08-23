import json
import os
import random
import numpy as np
from collections import defaultdict

import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A

import shapely
import shapely.geometry
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.ops import nearest_points


def compute_iou(a, b):
    a = np.array(a)
    poly1 = Polygon(a).convex_hull
    b = np.array(b)
    poly2 = Polygon(b).convex_hull

    union_poly = np.concatenate((a, b))
    if not poly1.intersects(poly2):
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # intersection area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                iou = 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def get_direction(start, end):
    vec = np.array(end) - np.array(start)
    _angle = 0
    #          90
    #      135    45
    #     180  .    0
    #      225   -45 
    #          270
    if vec[1] > 0:  # lng is postive
        _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90
    elif vec[1] < 0:
        _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90 + 180
    else:
        if np.sign(vec[0]) == 1:
            _angle = 90
        else:
            _angle = 270
    _angle = (360 - _angle + 90) % 360
    return _angle


def name_the_direction(_angle):
    if _angle > 337.5 or _angle < 22.5:
        return 'north'
    elif np.abs(_angle - 45) <= 22.5:
        return 'northeast'
    elif np.abs(_angle - 135) <= 22.5:
        return 'southeast'
    elif np.abs(_angle - 90) <= 22.5:
        return 'east'
    elif np.abs(_angle - 180) <= 22.5:
        return 'south'
    elif np.abs(_angle - 315) <= 22.5:
        return 'northwest'
    elif np.abs(_angle - 225) <= 22.5:
        return 'southwest'
    elif np.abs(_angle - 270) <= 22.5:
        return 'west'


class SimulatorAVDN(object):
    def __init__(self, input_batch, batch_size=4, dataset_dir=None):
        self.batch = input_batch
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir

        # for storing the complete remotesencing images and attention maps;
        self.map_batch = {}
        self.attention_map_batch = {}

        # load images and attention maps;
        self._load_images_maps()
        # build the augmentation transform;
        self.aug_transform = self._init_aug_transform()

    def _load_images_maps(self):
        used_map_names = []
        for i in range(len(self.batch)):
            # print(self.batch)
            used_map_names.append(self.batch[i]['map_name'])
            if not used_map_names[-1] in self.map_batch.keys():
                im = cv2.imread(os.path.join(self.dataset_dir, used_map_names[-1] + '.tif'), 1)

                lng_ratio = self.batch[i]['lng_ratio']
                lat_ratio = self.batch[i]['lat_ratio']
                im_resized = cv2.resize(im, (int(im.shape[1] * lng_ratio / lat_ratio), im.shape[0]), interpolation=cv2.INTER_AREA)  # ratio_all = lat_ratio
                self.map_batch[used_map_names[-1]] = im_resized

                attention_map = np.zeros((im_resized.shape[0], im_resized.shape[1], 3), np.uint8)
                for j in range(len(self.batch[i]['attention_list'])):
                    cv2.circle(attention_map, center=self.gps_to_img_coords(self.batch[i]['attention_list'][j][0], self.batch[i]),
                               radius=self.batch[i]['attention_list'][j][1],
                               color=(255, 255, 255),
                               thickness=-1)  # fill the circle
                self.attention_map_batch[used_map_names[-1]] = attention_map

        return used_map_names

    def _get_obs(self, corners=None, directions=None, t=None, not_in_train=False):
        obs = []

        for i in range(len(self.batch)):
            item = self.batch[i]
            if t is None:
                t_input = 0
            else:
                if t < len(item['gt_path_corners']):
                    t_input = t
                else:
                    t_input = len(item['gt_path_corners']) - 1
            if corners is None:
                view_area_corners = item['gt_path_corners'][t_input]
            else:
                view_area_corners = corners[i]

            # -------------------------------------------------------------------------------------- #
            # 1. generate view area
            # -------------------------------------------------------------------------------------- #
            width = 224
            height = 224
            dst_pts = np.array([[0, 0],
                                [width - 1, 0],
                                [width - 1, height - 1],
                                [0, height - 1]], dtype="float32")

            view_area_corners = np.array(view_area_corners)
            img_coord_view_area_corners = view_area_corners
            for xx in range(view_area_corners.shape[0]):
                img_coord_view_area_corners[xx] = self.gps_to_img_coords(view_area_corners[xx], item)
            img_coord_view_area_corners = np.array(img_coord_view_area_corners, dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(img_coord_view_area_corners, dst_pts)
            # directly warp the rotated rectangle to get the straightened rectangle
            im_view = cv2.warpPerspective(self.map_batch[item['map_name']], M, (width, height))
            gt_saliency = cv2.warpPerspective(self.attention_map_batch[item['map_name']], M, (width, height))
            gt_saliency = np.asarray(cv2.cvtColor(gt_saliency, cv2.COLOR_BGR2GRAY)) / 255

            # -------------------------------------------------------------------------------------- #
            # 2. image augmentation for current view
            # -------------------------------------------------------------------------------------- #
            if random.random() < 0.4 and not not_in_train:
                # print('auging auging.')
                im_view_rgb = cv2.cvtColor(im_view, cv2.COLOR_BGR2RGB)
                transformed_im_view = self.aug_transform(image=im_view_rgb)["image"]
                # save the aug image;
                transformed_im_view = cv2.cvtColor(transformed_im_view, cv2.COLOR_RGB2BGR)
            else:
                transformed_im_view = im_view

            # -------------------------------------------------------------------------------------- #
            # 3. prepare the destination bounding boxes for grounding.
            #    return the bbox in current image coordinates.
            # -------------------------------------------------------------------------------------- #
            # 1. destination coordinates;
            if not not_in_train:
                des_corners = np.array(item['destination'])
                img_coord_des_corners = des_corners.copy()
                for xx in range(des_corners.shape[0]):
                    # convert the gps coord to image coord;
                    img_coord_des_corners[xx] = self.gps_to_img_coords(des_corners[xx], item)

                # 2. check current view, if contains target then convert the coords;
                des_poly = shapely.geometry.Polygon(img_coord_des_corners)
                cur_poly = shapely.geometry.Polygon(img_coord_view_area_corners)

            if not not_in_train and cur_poly.contains(des_poly):
                # print('training gronding')
                # convert the destination's gps coords;
                aff_des_corners = cv2.perspectiveTransform(img_coord_des_corners.reshape([-1, 1, 2]), M)
                # aff_des_corners: [4,1,2] -> [4,2], np.unint
                aff_des_corners = aff_des_corners.reshape([4, 2]).astype(np.uint8)
                x_min, x_max, y_min, y_max = \
                    np.min(aff_des_corners[:, 0]), np.max(aff_des_corners[:, 0]), np.min(aff_des_corners[:, 1]), np.max(aff_des_corners[:, 1])
                target_xyxy = np.array([x_min, y_min, x_max, y_max])
                bbox_labels = 1.0
            else:
                # no targe, negtive label;
                target_xyxy = np.zeros([4, ])
                bbox_labels = 0.0

            # distance to the final point;
            current_pos = np.mean(view_area_corners, axis=0)
            gt_pos = np.mean(item['gt_path_corners'][-1], axis=0)
            dis_to_current = np.linalg.norm(gt_pos - current_pos)

            obs.append({
                'map_name': item['map_name'],
                'map_size': self.map_batch[item['map_name']].shape,
                'route_index': item['route_index'],

                'gps_botm_left': item['gps_botm_left'],
                'gps_top_right': item['gps_top_right'],

                'lng_ratio': item['lng_ratio'],
                'lat_ratio': item['lat_ratio'],
                'starting_angle': item['angle'],

                'current_view': transformed_im_view,
                'gt_saliency': gt_saliency,

                'gt_path_corners': item['gt_path_corners'],
                'view_area_corners': view_area_corners,

                'instructions': item['instructions'],
                'pre_dialogs': item['pre_dialogs'],

                # bbox target;
                'bbox_target_xyxy': target_xyxy,
                'bbox_labels': bbox_labels,
                'img_w': 224,
                'img_h': 224,

                'distance': dis_to_current,
            })

        return obs

    @staticmethod
    def _init_aug_transform():
        # image aug transforme
        aug_list = list()
        # image blur/
        aug_list.append(A.OneOf([A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                                 A.MotionBlur((3, 5), p=0.5)],
                                p=0.4))
        # image noise/
        aug_list.append(A.OneOf([A.GaussNoise(p=0.5),
                                 A.MultiplicativeNoise(p=0.5)],
                                p=0.4))
        # image light/
        aug_list.append(A.OneOf([A.RandomBrightnessContrast(0.18, 0.18, p=0.5),
                                 A.RandomGamma(gamma_limit=(90, 118), p=0.5)],
                                p=0.4))
        # image corrupt/
        aug_list.append(A.OneOf([A.PixelDropout(p=0.5),
                                 A.CoarseDropout(p=0.5)],
                                p=0.4))
        # image sharpen/ image boss/ image shear/
        aug_list.append(A.OneOf([A.Sharpen(alpha=(0.2, 0.4), lightness=(0.6, 1.2), p=0.5),
                                 A.Emboss(p=0.5), ],
                                p=0.3))
        # image random sunshine\fog\shadow
        aug_list.append(A.RandomShadow(p=0.125))
        # image affine/
        aug_list.append(A.PiecewiseAffine(scale=(0.01, 0.03), p=0.125))

        return A.Compose(aug_list)

    @staticmethod
    def gps_to_img_coords(gps, ob):
        gps_botm_left, gps_top_right = ob['gps_botm_left'], ob['gps_top_right']
        lng_ratio, lat_ratio = ob['lng_ratio'], ob['lat_ratio']

        return int(round((gps[1] - gps_botm_left[1]) / lat_ratio)), int(round((gps_top_right[0] - gps[0]) / lat_ratio))


class ANDHNavBatchMap(Dataset):
    def __init__(self, anno_dir, dataset_dir, splits, seed=0, ):
        self.dataset_dir = dataset_dir
        self.data_list = []

        for split in splits:
            data = json.load(open(os.path.join(anno_dir, '%s_data.json' % split)))
            for item in data:
                item['angle'] = round(item['angle']) % 360
                for i in range(len(item['gt_path_corners'])):
                    item['gt_path_corners'][i] = np.array(item['gt_path_corners'][i])

                item['instructions'] = item['instructions'].lower()
                item['pre_dialogs'] = ' '.join(item['pre_dialogs']).lower()

                self.data_list.append(item)

            print('%s loaded with %d instructions, using splits: %s' % (self.__class__.__name__, len(data), split))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    # Nav Evaluation
    @staticmethod
    def _eval_item(gt_path, gt_corners, path, corners, progress):
        def path_fid(exec_path, gt_path):
            div_sum = 0
            for i in range(len(exec_path)):
                poly = LineString(gt_path)
                point = Point(exec_path[i])
                p1, p2 = nearest_points(poly, point)
                div_sum += np.linalg.norm(np.array(p1.coords) - exec_path[i])
            return div_sum / len(exec_path)

        scores = {}
        scores['trajectory_lengths'] = np.sum([np.linalg.norm(a - b) for a, b in zip(path[:-1], path[1:])])
        scores['trajectory_lengths'] = scores['trajectory_lengths'] * 11.13 * 1e4
        gt_whole_lengths = np.sum([np.linalg.norm(a - b) for a, b in zip(gt_path[:-1], gt_path[1:])]) * 11.13 * 1e4
        gt_net_lengths = np.linalg.norm(gt_path[0] - gt_path[-1]) * 11.13 * 1e4

        scores['iou'] = progress[-1]  # same as compute_iou(corners[-1], gt_corners[-1]）

        scores['gp'] = gt_net_lengths - np.linalg.norm(path[-1] - gt_path[-2]) * 11.13 * 1e4
        scores['oracle_gp'] = gt_net_lengths - np.min([np.linalg.norm(path[x] - gt_path[-1]) for x in range(len(path))]) * 11.13 * 1e4

        scores['success'] = float(progress[-1] >= 0.4)
        _center = np.mean(gt_corners[-1], axis=0)
        _point = Point(_center)
        _poly = Polygon(np.array(corners[-1]))
        if not _poly.contains(_point):
            scores['success'] = float(0)

        _center = np.mean(corners[-1], axis=0)
        _point = Point(_center)
        _poly = Polygon(np.array(gt_corners[-1]))
        if not _poly.contains(_point):
            scores['success'] = float(0)

        scores['oracle_success'] = float(any(np.array(progress) > 0.4))
        scores['gt_length'] = gt_whole_lengths
        scores['spl'] = scores['success'] * gt_net_lengths / max(scores['trajectory_lengths'], gt_net_lengths, 0.01)

        return scores

    def eval_metrics(self, preds, human_att_eval=False):
        """ Evaluate each agent trajectory based on how close it got to the goal location
        the path contains [view_id, angle, vofv]"""
        # print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        if human_att_eval:
            for k in preds.keys():
                if 'human_att_performance' in preds[k].keys():
                    metrics['human_att_performance'] += preds[k]['human_att_performance']
                    nss = np.mean(preds[k]['nss'])
                    if nss == nss:
                        metrics['nss'].append(nss)
            metrics['human_att_performance'] = np.mean(metrics['human_att_performance'], axis=0)
            metrics['nss'] = np.mean(metrics['nss'])
            if metrics['nss'] == metrics['nss']:
                avg_metrics = {"HA_precision": metrics['human_att_performance'][0],
                               "HA_recall": metrics['human_att_performance'][0],
                               "nss": metrics['nss']}
            else:
                avg_metrics = {"HA_precision": 0, "HA_recall": 0, "nss": 0}
            return avg_metrics, metrics

        for k in preds.keys():
            item = preds[k]
            instr_id = item['instr_id']
            # print(instr_id)
            dia_number = 0
            if 'num_dia' in preds[k].keys():
                dia_number = preds[k]['num_dia']
            traj = [np.mean(x[0], axis=0) for x in item['path_corners']]  # x = (corners, directions)
            corners = [np.array(x[0]) for x in item['path_corners']]  # x = (corners, directions)
            progress = [x for x in item['gt_progress']]
            gt_corners = [np.array(x) for x in item['gt_path_corners']]
            gt_trajs = [np.mean(x, axis=0) for x in item['gt_path_corners']]

            traj_scores = self._eval_item(gt_trajs, gt_corners, traj, corners, progress)
            for k, v in traj_scores.items():
                if k == 'iou' and traj_scores['success']:
                    metrics[k].append(v)
                else:
                    metrics[k].append(v)

            if dia_number == 1:
                metrics['success_1'].append(traj_scores['success'])
                metrics['spl_1'].append(traj_scores['spl'])
                metrics['gp_1'].append(traj_scores['gp'])
            elif dia_number == 2:
                metrics['success_2'].append(traj_scores['success'])
                metrics['spl_2'].append(traj_scores['spl'])
                metrics['gp_2'].append(traj_scores['gp'])
            else:
                metrics['success_else'].append(traj_scores['success'])
                metrics['spl_else'].append(traj_scores['spl'])
                metrics['gp_else'].append(traj_scores['gp'])

            if traj_scores['trajectory_lengths'] > 150:
                metrics['success_long'].append(traj_scores['success'])
                metrics['spl_long'].append(traj_scores['spl'])
                metrics['gp_long'].append(traj_scores['gp'])
            else:
                metrics['success_short'].append(traj_scores['success'])
                metrics['spl_short'].append(traj_scores['spl'])
                metrics['gp_short'].append(traj_scores['gp'])
            metrics['instr_id'].append(instr_id)

        avg_metrics = {
            # 'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'gp': np.mean(metrics['gp']),
            'oracle_gp': np.mean(metrics['oracle_gp']),
            'gt_length': np.mean(metrics['gt_length']),
            'iou': np.mean(metrics['iou']),
        }
        if len(metrics['success_1']) != 0:
            avg_metrics['num_1'] = len(metrics['success_1'])
            avg_metrics['spl_1'] = np.mean(metrics['spl_1']) * 100
            avg_metrics['sr_1'] = np.mean(metrics['success_1']) * 100
            avg_metrics['gp_1'] = np.mean(metrics['gp_1'])

        if len(metrics['success_2']) != 0:
            avg_metrics['num_2'] = len(metrics['success_2'])
            avg_metrics['spl_2'] = np.mean(metrics['spl_2']) * 100
            avg_metrics['sr_2'] = np.mean(metrics['success_2']) * 100
            avg_metrics['gp_2'] = np.mean(metrics['gp_2'])

        if len(metrics['success_else']) != 0:
            avg_metrics['num_else'] = len(metrics['success_else'])
            avg_metrics['spl_else'] = np.mean(metrics['spl_else']) * 100
            avg_metrics['sr_else'] = np.mean(metrics['success_else']) * 100
            avg_metrics['gp_else'] = np.mean(metrics['gp_else'])

        return avg_metrics, metrics


def simple_cat_collate(data):
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]


if __name__ == '__main__':
    ds = ANDHNavBatchMap(anno_dir='../../datasets', dataset_dir=None, splits=['val_unseen'])
    dl = DataLoader(ds, batch_size=1, drop_last=False, collate_fn=simple_cat_collate)

    # print(ds.__getitem__(0))
    print(next(iter(dl)))
