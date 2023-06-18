import numpy as np

from plot import Plot


class Data(Plot):
    def init(self, image_path):
        Plot.__init__(self, image_path)

    def get_data(self):
        return self.Solar

    def get_corners(self):
        return self.selected_rect

    def set_corners(self, corners):
        self.selected_rect = corners

    def get_size(self):
        return self.Solar.shape

    def set_size(self, size):
        self.Solar = np.zeros(size)
