from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from rt_util.hand import HandIndex

class Realtime3DHandChart:
    def __init__(self, joint_scale):

        self.finger_indicies = [HandIndex.thumb_joints,
                                HandIndex.index_joints,
                                HandIndex.middle_joints,
                                HandIndex.ring_joints,
                                HandIndex.pinky_joints]

        self.finger_colors = ['r', 'b', 'c', 'm', 'g']
        self.joints = None
        self.axis_limit = 2 * joint_scale
        self.create()

    def create(self):
        self.figure = plt.figure(figsize = (6, 6))
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.autoscale(enable=True, axis='both', tight=True)


    def update(self, joints):

        self.ax.clear()
        lmt_val = self.axis_limit
        self.ax.set_xlim(-lmt_val, lmt_val)
        self.ax.set_ylim(-lmt_val, lmt_val)
        self.ax.set_zlim(0, lmt_val)

        for i, indicies in enumerate(self.finger_indicies):
            indicies = [HandIndex.wrist] + indicies
            x = joints[indicies, 0]
            y = joints[indicies, 1]
            z = joints[indicies, 2]

            color = self.finger_colors[i]
            self.ax.plot(x, y, z, c=color, marker='o')

        plt.draw()
        plt.pause(0.001)

    def show(self):
        plt.ion()
        plt.show()
