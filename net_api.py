"""DeepFisheye API
This API module makes easier to use the DeepFisheye model for realtime application.
"""
import pathlib
import sys

deepfisheye_module_path = pathlib.Path(__file__).parents[0] / "DeepFisheyeNet"
print(deepfisheye_module_path)
sys.path.append(str(deepfisheye_module_path))

from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

from DeepFisheyeNet.option.options import Options
from DeepFisheyeNet.model.pipeline.deep_fisheye_pipeline import DeepFisheyePipeline
from rt_util.hand import HandIndex
from rt_util.setting import load_setting


def get_model(setting):
    options = get_default_options(setting)
    return DeepFisheyeModel(options)

def get_default_options(setting):
    options = Options()
    options.initialize_with_defaults()
    apply_settings(options, setting['deepfisheye'])
    options.parse()
    return options

def apply_settings(options, setting):
    options.general.img_size = setting['img_size']
    options.general.gpu_ids = setting['gpu_ids']
    options.hpe.mode = setting['mode']
    options.hpe.img_size = setting['img_size']
    options.hpe.input_channel = 3
    options.hpe.norm_type = setting['norm_type']
    options.hpe.joint_scale = setting['joint_scale']
    options.hpe.resume_train_epoch = 0

    options.general.pipeline_pretrained = setting['weight_file']
    options.general.img_size = setting['img_size']

    options.pix2depth.mode = setting['mode']
    options.pix2depth.net = 'resnet_3blocks'
    options.pix2depth.norm_type = setting['norm_type']
    options.pix2depth.img_size = setting['img_size']
    options.pix2depth.output_nc = 1

    options.hpe.mode = setting['mode']
    options.hpe.input_channel = 4
    options.hpe.img_size = setting['img_size']

class DeepFisheyeModel:
    def __init__(self, options):
        self.options = options
        self.model = self._create_model(options)
        self.transform = self._create_transform(options)
        self.hand_size = options.hpe.joint_scale

    def _create_model(self, options):
        model = DeepFisheyePipeline(options)
        model.setup()
        return model

    def _create_transform(self, options):
        transform_list = []
        img_size = options.general.img_size
        transform_list += [transforms.Resize((img_size, img_size))]
        transform_list += [transforms.ToTensor()]
        return transforms.Compose(transform_list)

    def estimate_pose(self, cv_img):
        input = DeepFisheyeInput(cv_img)
        img = input.get_pil_image()

        img = self.transform(img)
        model_input = self._convert_to_model_input(img)
        self.model.set_input(model_input)
        self.model.forward()

        results = self.model.get_detached_current_results()
        output = DeepFisheyeOutput(results)
        return output

    def _convert_to_model_input(self, img):
        img = img.unsqueeze(0)
        input = {'fish': img}
        return input

class DeepFisheyeInput:
    def __init__(self, cv_img):
        assert self._is_opencv_img(cv_img)

        self.cv_img = cv_img

    def _is_opencv_img(self, img):
        # (H, W, C)
        return (len(img.shape) == 3) and img.shape[2] <=3

    def get_pil_image(self):
        if hasattr(self, 'pil_img'):
            return self.pil_img

        self.pil_img = self._convert_to_pil(self.cv_img)
        return self.pil_img

    def _convert_to_pil(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        return pil_img

class DeepFisheyeOutput:
    fingertip_id = HandIndex.fingertip_joints
    def __init__(self, results):
        self.results = results

    def _normalize_size(self, results):
        joints = results['joint'].squeeze()

    def get_predicted_pose(self, only_fingertip = False):
        pos = self._extract_pose(self.results, only_fingertip)
        return pos

    def _extract_pose(self, results, only_fingertip):
        joints = results['joint'].squeeze().numpy()
        if only_fingertip:
            return self._extract_only_fingertips(joints)
        return joints

    def get_heatmap(self, only_fingertip = False):
        heatmaps = self._extract_heatmaps(self.results, only_fingertip)
        return heatmaps

    def get_heatmap_img(self, only_fingertip = False):
        heatmaps = self.get_heatmap(only_fingertip)
        merged = heatmaps.sum(axis = 0)
        merged = np.clip(merged, 0, 1)
        cv2_img = self._convert_to_cv2_image(merged)
        colored = cv2.applyColorMap(cv2_img, cv2.COLORMAP_JET)

        return colored

    def _extract_heatmaps(self, results, only_fingertip):
        heatmaps = results['heatmap'].squeeze().numpy()
        if only_fingertip:
            return self._extract_only_fingertips(heatmaps)

        return heatmaps

    def get_fingertips_and_wrist_heatmap_img(self):
        heatmaps = self._extract_fingertips_and_wrist_heatmaps(self.results)
        merged = heatmaps.sum(axis = 0)
        merged = np.clip(merged, 0, 1)
        cv2_img = self._convert_to_cv2_image(merged)
        colored = cv2.applyColorMap(cv2_img, cv2.COLORMAP_JET)

        return colored

    def _extract_fingertips_and_wrist_heatmaps(self, results):
        heatmaps = results['heatmap'].squeeze().numpy()
        return self._extract_fingertips_and_wrist(heatmaps)

        return heatmaps

    def _convert_to_cv2_image(self, np_from_pil):
        np_from_pil = np.clip(np_from_pil, 0, 1)
        cv2_img = np_from_pil * 255
        cv2_img = cv2_img.astype(np.uint8)

        if len(cv2_img.shape) > 2:
            # pil and cv2 has different axis scheme
            cv2_img = np.swapaxes(cv2_img, 0, 2)
            cv2_img = np.swapaxes(cv2_img, 0, 1)
        return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    def get_fake_fish_depth_img(self):
        if not 'fake_fish_depth' in self.results:
            return None
        fake_fish = self._extract_fake_fish_depth(self.results)
        cv2_img = self._convert_to_cv2_image(fake_fish)
        colored = cv2.applyColorMap(cv2_img, cv2.COLORMAP_JET)
        return colored

    def _extract_fake_fish_depth(self, results):
        fake_fish = results['fake_fish_depth'].squeeze().numpy()
        return fake_fish

    def get_segmented_hand_img(self):
        if not 'hand_pix' in self.results:
            return None

        hand_pix = self._extract_hand_pix(self.results)
        cv2_img = self._convert_to_cv2_image(hand_pix)
        return cv2_img

    def _extract_hand_pix(self, results):
        return results['hand_pix'].squeeze().numpy()

    def get_prob(self, only_fingertip = False):

        heatmaps = self._extract_heatmaps(self.results, only_fingertip)

        flat_heatmaps = self._flatten_heatmaps(heatmaps)
        return flat_heatmaps.max(axis = 1)

    def _flatten_heatmaps(self, heatmaps):
        num_joints = heatmaps.shape[0]
        return heatmaps.reshape(num_joints, -1)

    def get_max_prob_2d_location(self, only_fingertip):
        heatmaps = self._extract_heatmaps(self.results, only_fingertip)
        locs = []
        for heatmap in heatmaps:
            loc = self._argmax_2d(heatmap)
            locs.append(loc)
        return locs

    def _argmax_2d(self, mat):
        return np.unravel_index(mat.argmax(), mat.shape)

    def _extract_only_fingertips(self, np_mat):
        return np_mat[self.fingertip_id]

    def _extract_fingertips_and_wrist(self, np_mat):
        idx = [HandIndex.wrist] + HandIndex.fingertip_joints
        return np_mat[idx]

    def set_true_pose(self, true_joints):
        self.true_pose = true_joints

    def get_true_pose(self, only_fingertip = False):
        if only_fingertip:
            return self._extract_only_fingertips(self.true_pose)
        return self.true_pose

    def get_hand_size(self):
        center_idx = HandIndex.center
        wrist_idx = HandIndex.wrist

        joint = self._extract_pose(self.results, False)
        diff = joint[center_idx, :] - joint[wrist_idx, :]
        return np.linalg.norm(diff)
