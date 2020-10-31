# DeepFisheye Realtime
This code is for running pretrained model of [DeepFisheyeNet](https://github.com/KAIST-HCIL/DeepFisheyeNet).
To find out more about DeepFisheye, please visit the [project page](http://kwpark.io/deepfisheye).

## How to use

### Install prerequisites
#### 1. Install Spinnaker
The file can be downloaded from [FLIR](https://www.flir.com/products/spinnaker-sdk/).
#### 2. Install PySpin
PySpin is Python API of Spinnaker. It is possible to access an FLIR camera with this library. Install instructions and the library file are in the Spinnaker.
#### 3. Install pip packages
```shell
$ pip install -r requirements.txt
```
#### 4. Download the pretrained weight file
Download the file from [Google Drive](https://drive.google.com/file/d//1C_kbaw1Ull4D_JHgDkhrLwdjCITzj-8E/view?usp=sharing) and put the file in the root directory.

## Apparatus

DeepFisheye is tested with the following apparatus.
- [FLIR Chameleon3 (FLIR CM3-U3-13Y3C-CS)](https://www.flir.com/products/chameleon3-usb3/)
- [Fisheye lens (M12) (from AliExpress, No model ID)](https://ko.aliexpress.com/item/32795299264.html?spm=a2g0s.9042311.0.0.618c4c4dSzOV1a)
- [FLIR S Mount (M12) Lens Holder](https://www.flir.com/products/s-mount-m12-lens-holder/)

It is possible to use different cameras. However, in that case, there are some issues to take care about.

### Motion blur
Motion blur makes harder for the network to recognize hands. We tried chip USB cameras, but they had too much motion blur. It is why we used the machine vision camera.

### Camera calibration
If the same lens (and the same lens mount) is used, then our camera parameters in ```setting.yaml``` would work fine. However, different apparatus is used, then camera calibration has to be done.

The calibration can be done with [OCamCalib](https://sites.google.com/site/scarabotix/ocamcalib-toolbox). It is really great tool. Here is a rough process of camera calibration.

#### OCamCalib calibration
1. Take 8 to 10 images with the camera.
2. Crop the images to square images. Do not randomly cut the images. Carefully chose center and width of the cropping. And set ```crop_width_center```, ```crop_height_center```, ```width``` , and ```height``` in ```setting.yaml```. This code cuts images from the camera with these parameters.
3. Get camera parameters from OCamCalib.
4. Put the camera parameters in ```setting.yaml``` (```z_calib``` and ```affine_calib```).

## Tips

### Hand size
The outputs of DeepFisheyeNet work is not in correct scale. Therefore, ```joint_scale``` should be set correctly in ```setting.yaml```. The length between a user's wrist and Metacarpophalangeal joint of middle finger (hand size) in milli meter should be put in ```joint_scale```. If you don't know your hand size, ```70``` is a good value to start. Try adjusting this value.

### Filter
The outputs of DeepFisheyeNet cannot be directly used for interaction application, because they are too noisy. Therefore, we added [One Euro filters](https://cristal.univ-lille.fr/~casiez/1euro/) for the demo application (not in experiment).

However, we did not include filters in this code. But, we share the parameters that we used.
```python
# One Euro filter params
min_cutoff = 1
beta = 0.05
d_cutoff = 1.0
```
## Contact
If you have any questions, please contact [Keunwoo Park](http://kwpark.io) or add an issue.
