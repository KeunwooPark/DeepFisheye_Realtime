import cv2
import os
import time
import PySpin
import numpy as np

def create_camera(setting):
    return FlirCamera(setting['camera']['exposure'])

class FlirCamera:
    def __init__(self, exposure = -1):
        """FlirCamera

        Note about exposure.
        - if exposure is < 0, it is set automatically.
        - in microseconds
        """
        self.cam = None
        self.exposure = exposure

        self._set_camera()

    def _set_camera(self):
        self._init_flir_system()
        self.nodemap_tldevice, self.nodemap = self._init_cam()
        self._set_cam_parameters(self.nodemap, self.nodemap_tldevice)
        self.cam.BeginAcquisition()

    def _init_flir_system(self):
        self.system = PySpin.System.GetInstance()

        # Get current library version
        version = self.system.GetLibraryVersion()
        print('Spinnake Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        # Retrieve list of cameras from the system
        self.cam_list = self.system.GetCameras()
        num_cameras = self.cam_list.GetSize()
        print('Number of cameras detected: %d' % num_cameras)

        assert num_cameras == 1, "the number of camera shoulde be one"

        self.cam = self.cam_list[0]

    def _init_cam(self):
        cam = self.cam
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        cam.Init()
        nodemap = cam.GetNodeMap()

        return nodemap_tldevice, nodemap

    def _set_cam_parameters(self, nodemap, nodemap_tldevice):

        self._set_bufferhandling_mode_to_newest_only()
        self._set_pixel_format_to_rgb8(nodemap)
        self._set_acquisition_mode(nodemap)
        self._set_exposure(nodemap, self.exposure)

    def _set_bufferhandling_mode_to_newest_only(self):
        cam = self.cam

        sNodemap = cam.GetTLStreamNodeMap()
        self._set_node_to_entry(sNodemap, 'StreamBufferHandlingMode', 'NewestOnly')

    def _set_pixel_format_to_rgb8(self, nodemap):
        self._set_node_to_entry(nodemap, 'PixelFormat', 'RGB8')

    def _set_acquisition_mode(self, nodemap):
        self._set_node_to_entry(nodemap, 'AcquisitionMode', 'Continuous')

    def _set_exposure(self, nodemap, exposure):
        if exposure < 0:
            self._set_exposure_to_auto(nodemap)
        else:
            self._turn_off_exposure_auto(nodemap)
            self._set_exposure_time(nodemap, exposure)

    def _set_exposure_to_auto(self, nodemap):
        self._set_node_to_entry(nodemap, 'ExposureAuto', 'Continuous')

    def _turn_off_exposure_auto(self, nodemap):
        self._set_node_to_entry(nodemap, 'ExposureAuto', 'Off')

    def _set_exposure_time(self, nodemap, exposure_time):
        """Following QuickSpin Syntax."""
        cam = self.cam

        assert self.cam.ExposureTime.GetAccessMode() == PySpin.RW

        exposure_time = int(round(exposure_time))
        max_exposure_time = self.cam.ExposureTime.GetMax()
        if exposure_time > max_exposure_time:
            print("exposure time is reset to max_exposure_time {}".format(exposure_time))
            exposure_time = max_exposure_time

        cam.ExposureTime.SetValue(exposure_time)

    def _set_node_to_entry(self, nodemap, node_name, entry_name):
        node = self._get_writable_node(nodemap, node_name)
        entry = self._get_readable_entry(node, entry_name)
        entry_value = entry.GetValue()
        node.SetIntValue(entry_value)

    def _get_writable_node(self, nodemap, node_name):
        node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
        self._check_is_writable(node, node_name)
        return node

    def _get_readable_entry(self, node, entry_name):
        node_entry = node.GetEntryByName(entry_name)
        self._check_is_readable(node_entry, entry_name)
        return node_entry

    def _check_is_writable(self, pyspin_obj, desc):
        assert PySpin.IsAvailable(pyspin_obj) and PySpin.IsWritable(pyspin_obj), "{} is not writable".format(desc)

    def _check_is_readable(self, pyspin_obj, desc):
        assert PySpin.IsAvailable(pyspin_obj) and PySpin.IsReadable(pyspin_obj), "{} is not readable".format(desc)

    def get_frame(self):
        assert self.cam is not None
        cam = self.cam

        image_result = cam.GetNextImage()

        if image_result.IsIncomplete():
            print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
            return
        else:
            frame = image_result.GetNDArray()

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    def release(self):
        #self.timer.print_mean_interval()
        self.timer.print_fps()
        self.cam.EndAcquisition()
        self.cam.DeInit()
        self.cam_list.Clear()
        self.system.ReleaseInstance()
