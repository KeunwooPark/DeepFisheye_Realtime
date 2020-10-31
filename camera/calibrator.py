import cv2
import pickle
import numpy as np
from PIL import Image
import pathlib
from camera.fisheye import fisheye_func, fisheye_func_inv
from rt_util.setting import load_setting

root_path = pathlib.Path(__file__).parents[1];
dir_path = root_path / "camera";

class FisheyeCalibrator:
    valid_fisheye_types = ['stereographic', 'equidistant', 'equisolid', 'orthographic']

    def __init__(self, fisheye_type = 'equidistant'):
        self.parse_setting()
        assert fisheye_type in self.valid_fisheye_types, "not valid fishseye type"

        coeffs = self.params['image']
        self.img_height = int(coeffs['height'])
        self.img_width = int(coeffs['width'])
        self.crop_width_center = int(coeffs['crop_width_center'])
        self.crop_height_center = int(coeffs['crop_height_center'])
        self.fisheye_type = fisheye_type
        self.is_remap_mat_set = False

        self.set_f()

    def set_f(self):
        """ Set appropriate 'f' so that end of the image is PI/2"""
        max_dist = min((self.img_height, self.img_width)) / 2
        self.f = max_dist / fisheye_func[self.fisheye_type](np.pi / 2) # f*fisheye_func(pi/2) should be max_dist

    def set_remap_mats(self):
        if self._does_calib_mat_exists():
            self._load_saved_calib_mat()
            print("Calibrator use existing calibration matrix.")
        else:
            self._create_calib_mat()
            self._save_calib_mat()

        self.is_remap_mat_set = True

    def _does_calib_mat_exists(self):
        calib_mat_file_path = self._get_calib_mat_file_path()
        return calib_mat_file_path.exists()

    def _load_saved_calib_mat(self):
        calib_mat_file_path = self._get_calib_mat_file_path()
        calib_mat = pickle.load(open(str(calib_mat_file_path), 'rb'))
        self.a_uu, self.a_vv, self.r_uu, self.r_vv = calib_mat

    def _get_calib_mat_file_path(self):
        return dir_path / 'calib_mat.pkl'

    def _create_calib_mat(self):
        self.a_uu, self.a_vv = self.create_affine_calib_mat()
        self.r_uu, self.r_vv = self.create_radial_calib_mat()

    def _save_calib_mat(self):
        calib_mat = (self.a_uu, self.a_vv, self.r_uu, self.r_vv)
        calib_mat_file_name = str(self._get_calib_mat_file_path())
        pickle.dump(calib_mat, open(calib_mat_file_name, 'wb'))

    def calibrate(self, img, radial_calibration = True):
        assert self.is_remap_mat_set, "call 'set_remap_mats' before calibration"

        cal_img = self._crop(img, self.img_height, self.img_width, self.crop_height_center, self.crop_width_center)
        cal_img = cv2.remap(cal_img, self.a_uu, self.a_vv, cv2.INTER_LINEAR)
        if radial_calibration:
            cal_img = cv2.remap(cal_img, self.r_uu, self.r_vv, cv2.INTER_LINEAR)
        res = self.post_process(cal_img)
        return res

    def _crop(self, frame, height, width, h_center, w_center):
        f_height, f_width, channel = frame.shape
        assert f_height >= height and f_width >= width, "crop size ({}, {}) is bigger than the frame size ({}, {})".format(height, width, f_height, f_width)

        h_from = int(h_center - height / 2)
        w_from = int(w_center - width / 2)
        cropped = frame[h_from: h_from+height, w_from: w_from + width, :]

        return cropped

    def parse_setting(self):
        setting = load_setting()
        self.params = setting['camera']['calibration']

    def create_affine_calib_mat(self):
        coeffs = self.params['image']
        calib_center_w = coeffs['calib_center_w']
        calib_center_h = coeffs['calib_center_h']

        xx, yy = self.get_xy_meshgrid()

        uu = xx - calib_center_w
        vv = yy - calib_center_h

        uu, vv = self.affine_calibration(uu, vv)

        uu = np.squeeze(uu).astype(np.float32)
        vv = np.squeeze(vv).astype(np.float32)

        uu = uu + (self.img_width / 2)
        vv = vv + (self.img_height / 2)
        return uu, vv

    def affine_calibration(self, u, v):
        coeffs = self.params['affine_calib']
        c = coeffs['c']
        d = coeffs['d']
        e = coeffs['e']

        A = np.array([[c, d], [e, 1]], dtype=np.float64)
        inv_A = np.linalg.inv(A)
        UV = np.array([[u],[v]], dtype=np.float64)
        _uv = np.matmul(UV.T, A.T)
        _uv = _uv.T

        return _uv

    def get_xy_meshgrid_centered(self):
        xx, yy = self.get_xy_meshgrid()

        _xx = xx - (self.img_width / 2)
        _yy = yy - (self.img_height / 2)

        return _xx, _yy

    def get_xy_meshgrid(self):
        x_values = np.array(list(range(self.img_width)), dtype=np.float32)
        y_values = np.array(list(range(self.img_height)), dtype=np.float32)
        xx, yy = np.meshgrid(x_values, y_values)

        return xx, yy

    def create_radial_calib_mat(self):
        print("Generating radial calibration matrix... This takes some time. Please be patient.")
        rho =  self.get_rho_mat()
        phi = self.get_phi_mat()
        uu = rho * np.cos(phi)
        vv = rho * np.sin(phi)
        uu = uu + (self.img_width / 2)
        vv = vv + (self.img_height / 2)

        uu = np.nan_to_num(uu, 0).astype(np.float32)
        vv = np.nan_to_num(vv, 0).astype(np.float32)
        return uu, vv

    def get_rho_mat(self):
        xx, yy = self.get_xy_meshgrid_centered()

        coeff = self.cal_coeff(xx, yy)
        rho = self.solve_rho_equation(coeff)
        return rho

    def cal_coeff(self, u, v):
        inv_h = fisheye_func_inv[self.fisheye_type]

        r = np.sqrt(u**2 + v**2)
        coeff = np.tan(-inv_h(r/self.f))
        return coeff

    def solve_rho_equation(self, coeff):
        coeff_shape = coeff.shape
        poly_coeffs = self.params['z_calib']
        coeff = coeff.flatten()

        a0 = poly_coeffs['a0']
        a1 = poly_coeffs['a1'] # this is 0
        a2 = poly_coeffs['a2']
        a3 = poly_coeffs['a3']
        a4 = poly_coeffs['a4']

        result = []
        for c in coeff:
            if np.isnan(c):
                result.append(np.nan)
                continue
            poly_coeff = [a4*c, a3*c, a2*c, a1 - 1, a0*c]
            roots = np.roots(poly_coeff)
            answer = self.select_positive_real_number(roots)
            result.append(answer)

        result = np.array(result)

        return np.reshape(result, coeff_shape)

    def select_positive_real_number(self, roots):
        is_real = np.isreal(roots)
        is_positive = roots >= 0

        selected = np.logical_and(is_real, is_positive)
        selected_idx = np.argwhere(selected)
        if len(selected_idx) == 0:
            return np.nan
        selected_idx = selected_idx[0]
        if len(selected_idx) > 1:
            assert False, "Something is wrong with solving rho equation"

        return roots[selected_idx[0]].real

    def get_phi_mat(self):
        xx, yy = self.get_xy_meshgrid_centered()
        phi = np.arctan2(yy, xx)
        return phi

    def z_func(self, r):
        r2 = np.power(r, 2)
        r3 = np.power(r, 3)
        r4 = np.power(r, 4)

        coeffs = self.params['z_calib']
        return coeffs['a0'] + \
                coeffs['a1']*r + \
                coeffs['a2']*r2 + \
                coeffs['a3']*r3 + \
                coeffs['a4']*r4

    def post_process(self, img):
        coeffs = self.params['image']
        img_height = int(coeffs['height'])
        img_width = int(coeffs['width'])

        masked = self.mask_img(img, img_width, img_height)
        #cropped = self.center_crop_square(masked, img_width, img_height)
        return masked

    def mask_img(self, img, img_width, img_height):
        mask = self.get_center_mask(img_width, img_height)
        res = cv2.bitwise_and(img, img, mask = mask)
        return res

    def get_center_mask(self, img_width, img_height):
        mask = np.zeros((img_height, img_width), dtype=np.int8)

        img_center_x = int(img_width / 2)
        img_center_y = int(img_height / 2)
        f = np.min([img_center_x, img_center_y])

        cv2.circle(mask, (img_center_x, img_center_y), f, 255, -1)
        return mask

    def center_crop_square(self, img, img_width, img_height):
        length = np.min([img_width, img_height])

        up_left_x = int((img_width - length) / 2)
        up_left_y = int((img_height - length) / 2)

        # x, y flipped!
        cropped = img[up_left_y:up_left_y+length, up_left_x:up_left_x+length]
        return cropped
