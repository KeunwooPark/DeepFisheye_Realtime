import net_api as api
from camera.flir import create_camera
from camera.calibrator import FisheyeCalibrator
from camera.image_enhance import enhance_image
import cv2
import numpy as np
from rt_util.setting import load_setting
from rt_util.chart import Realtime3DHandChart

def main():

    setting = load_setting()
    camfing_model = api.get_model(setting)

    camera = create_camera(setting)
    calibrator = FisheyeCalibrator()
    calibrator.set_remap_mats()
    chart = Realtime3DHandChart(setting['deepfisheye']['joint_scale'])
    chart.show()

    while True:

        img = camera.get_frame()
        img = calibrator.calibrate(img)
        img = enhance_image(img)
        output = camfing_model.estimate_pose(img)

        heatmap_img = output.get_fingertips_and_wrist_heatmap_img()
        fake_fish_depth_img = output.get_fake_fish_depth_img()
        segmented_hand_img = output.get_segmented_hand_img()
        full_joints = output.get_predicted_pose(only_fingertip = False)

        chart.update(full_joints)

        images = [img, fake_fish_depth_img, segmented_hand_img, heatmap_img]
        images = resize_images(images, img.shape[:-1])
        merged_image = merge_to_single_image(images)

        cv2.imshow('img', merged_image)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break


def resize_images(images, img_shape):
    resized = []
    for image in images:
        resized.append(resize_img_empty_if_None(image, img_shape))

    return resized

def resize_img_empty_if_None(img, img_shape):
    if img is None:
        return get_empty_img(img_shape)

    return cv2.resize(img, img_shape)

def get_empty_img(img_shape):
    return np.zeros(img_shape)

def merge_to_single_image(images):
    return np.concatenate(images, axis = 1)

if __name__ == "__main__":
    main()
