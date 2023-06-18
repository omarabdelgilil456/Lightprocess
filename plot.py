import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
import scipy.ndimage
from matplotlib.widgets import RectangleSelector


class Plot:
    def __init__(self, image_path):
        self.t = np.linspace(0, 31.3, 100)
        self.f = np.linspace(0, 1000, 1000)
        self.a = np.exp(-np.abs(self.f-200)/200)[:, None] * np.random.rand(self.t.size)
        self.flim = (self.f.min(), self.f.max())
        self.tlim = (self.t.min(), self.t.max())
        self.selected_rect = []
        self.Solar = plt.imread(image_path)

        self.gs = gridspec.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1])
        self.ax = plt.subplot(self.gs[0,1])
        self.axl = plt.subplot(self.gs[0,0], sharey=self.ax)
        self.axb = plt.subplot(self.gs[1,1], sharex=self.ax)

        self.ax.set_title("Image of Spectrum")
        self.im = self.ax.imshow(self.Solar, origin='lower', aspect='auto',cmap="gray")

        self.axl.plot(self.Solar.sum(1),np.arange(self.Solar.shape[0]))
        self.axl.set_title("Sum by Rows")

        self.SolarSpecNormalized = self.Solar.sum(0)/self.Solar.sum(0).max()
        self.axb.plot(np.arange(self.Solar.shape[1]), self.SolarSpecNormalized)
        self.axb.set_title("Sum by Columns")

        self.windowsize = 75
        self.window = np.hamming(self.windowsize)
        self.filteredSolarConv = scipy.ndimage.convolve1d(self.Solar.sum(0), self.window)

        self.fSolConvNormalized = self.filteredSolarConv/self.filteredSolarConv.max()
        self.lineFinderSpectrum = np.divide(self.SolarSpecNormalized, self.fSolConvNormalized)
        self.rs = RectangleSelector(self.ax, self.on_rect_select, drawtype='box', useblit=True, button=[1], minspanx=5,
                                    minspany=5, spancoords='pixels', interactive=True, rectprops=dict(fill=None, alpha=1))

    def onpress(self, event):
        if event.inaxes == self.ax:
            x0, y0 = event.xdata, event.ydata
            self.rect.set_width(0)

    def on_rect_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.selected_rect = [int(min(x1, x2)), int(min(y1, y2)), int(abs(x1-x2)), int(abs(y1-y2))]

        self.show_data_popup()

    def show_data_popup(self):
        Solar = self.Solar[self.selected_rect[1]:self.selected_rect[1] + self.selected_rect[3],
                        self.selected_rect[0]:self.selected_rect[0] + self.selected_rect[2]]
        data_fig, data_ax = plt.subplots()
        data_ax.imshow(Solar, origin='lower', aspect='auto', cmap="gray")
        data_ax.set_title("Selected Data")

        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[3, 1])
        # main display
        ax = plt.subplot(gs[0, 1])

        # left to the main display
        axl = plt.subplot(gs[0, 0], sharey=ax)

        # at bottom of m,ain display
        axb = plt.subplot(gs[1, 1], sharex=ax)

        # fill main display with image
        ax.set_title("Image of Spectrum")
        ax.imshow(Solar, origin='lower', aspect='auto', cmap="gray")

        # fill left display with sum of rows
        axl.plot(Solar.sum(1), np.arange(Solar.shape[0]))
        axl.set_title("Sum by Rows")

        # do some math
        SolarSpecNormalized = Solar.sum(0) / Solar.sum(0).max()

        # fill bottom display with sum over columns
        axb.plot(np.arange(Solar.shape[1]), SolarSpecNormalized)
        axb.set_title("Sum by Columns")

        # do some signal processing (a low pass filter) on the spectrum
        windowsize = 75
        window = np.hamming(windowsize)
        filteredSolarConv = scipy.ndimage.convolve1d(Solar.sum(0), window)

        fSolConvNormalized = filteredSolarConv / filteredSolarConv.max()
        lineFinderSpectrum = np.divide(SolarSpecNormalized, fSolConvNormalized)

        # plot processed spectra.
        # plt.plot(lineFinderSpectrum.mean() * np.ones(lineFinderSpectrum.shape))
        # plt.plot( lineFinderSpectrum)

        plt.show()

