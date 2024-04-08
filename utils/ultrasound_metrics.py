import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import cv2

class Metrics(object):
    def __init__(self, dx, dz, gain, r, FWHM_threshold, color_limit):
        self.dx = dx
        self.dz = dz
        self.gain = gain
        self.r = r
        self.FWHM_threshold = FWHM_threshold
        self.color_limit = color_limit


    def get_Contrast_CNR(self, test_data, lesion, background):
        mu_lesion = np.mean(test_data[lesion[0]: lesion[2], lesion[1]: lesion[3]])
        std_lesion = np.std(test_data[lesion[0]: lesion[2], lesion[1]: lesion[3]])
        
        mu_bg = np.mean(test_data[background[0]: background[2], background[1]: background[3]])
        std_bg = np.std(test_data[background[0]: background[2], background[1]: background[3]])
        
        contrast = 20 * np.log10(np.abs(mu_lesion / mu_bg))
        CNR = 20 * np.log10(np.abs(mu_lesion - mu_bg) / np.sqrt(0.5 * (std_lesion ** 2 + std_bg ** 2)))
        
        return contrast, CNR

    def get_lateral_FWHM(self, test_data, FWHM_area):
        h, w = test_data.shape
        
        # print(np.argmax(test_data) // 96, np.argmax(test_data) % 96)
        
        # xx = self.dx * np.arange(self.r * w) / self.r
        
        # indz, indx = np.unravel_index(np.argmax(test_data[FWHM_area[0]: FWHM_area[2], :], axis=None), test_data[FWHM_area[0]: FWHM_area[2], :].shape)
        indz, indx = np.unravel_index(np.argmax(test_data[FWHM_area[0]: FWHM_area[2], FWHM_area[1]: FWHM_area[3]], axis=None), test_data[FWHM_area[0]: FWHM_area[2], FWHM_area[1]: FWHM_area[3]].shape)
        indz = indz + FWHM_area[0]
        indx = indx + FWHM_area[1]
        
        xx = self.dx * np.arange(FWHM_area[1] * self.r, FWHM_area[3] * self.r) / self.r
        xx0 = self.dx * np.arange(FWHM_area[1], FWHM_area[3])
        temp_data = test_data[indz, FWHM_area[1]: FWHM_area[3]]
        linear_func = interpolate.interp1d(xx0, 20 * np.log10(temp_data, where=temp_data>0), kind='linear', fill_value='extrapolate') 
        prof = linear_func(xx)
        
        
        '''
        xx0 = self.dx * np.arange(w)
        linear_func = interpolate.interp1d(xx0, 20 * np.log10(test_data[indz, :], where=test_data[indz, :]>0), kind='linear', fill_value='extrapolate') 
        prof = linear_func(xx)
        '''
        
        prof_max_index = np.argmax(prof)
        prof_max = prof[prof_max_index]
        
        prof = prof - prof_max
        
        left = 0
        for x in range(prof_max_index, 0, -1):
            if prof[x] > self.FWHM_threshold and prof[x-1] < self.FWHM_threshold:
                if prof[x] - self.FWHM_threshold > self.FWHM_threshold - prof[x-1]:
                    left = x - 1
                else:
                    left = x
        
        right = 0
        # for x in range(prof_max_index, self.r*w-1):
        for x in range(prof_max_index, len(prof) - 1):
            if prof[x] > self.FWHM_threshold and prof[x+1] < self.FWHM_threshold:
                if prof[x] - self.FWHM_threshold > self.FWHM_threshold - prof[x+1]:
                    right = x + 1
                else:
                    right = x
        
        lateral_FWHM = (right - left) * self.dx / self.r
        
        return lateral_FWHM
    
    def get_axial_FWHM(self, test_data, FWHM_area):
        h, w = test_data.shape
        
        # zz = self.dz * np.arange(self.r * h) / self.r
        
        indz, indx = np.unravel_index(np.argmax(test_data[FWHM_area[0]: FWHM_area[2], FWHM_area[1]: FWHM_area[3]], axis=None), test_data[FWHM_area[0]: FWHM_area[2], FWHM_area[1]: FWHM_area[3]].shape)
        indz = indz + FWHM_area[0]
        indx = indx + FWHM_area[1]
        
        zz = self.dz * np.arange(FWHM_area[0] * self.r, FWHM_area[2] * self.r) / self.r
        zz0 = self.dz * np.arange(FWHM_area[0], FWHM_area[2])
        temp_data = test_data[FWHM_area[0]: FWHM_area[2], indx]
        linear_func = interpolate.interp1d(zz0, 20 * np.log10(temp_data, where=temp_data>0), kind='linear', fill_value='extrapolate') 
        prof = linear_func(zz)
        
        
        
        '''
        zz0 = self.dz * np.arange(h)
        linear_func = interpolate.interp1d(zz0, 20 * np.log10(test_data[:, indx], where=test_data[:, indx]>0), kind='linear', fill_value='extrapolate') 
        prof = linear_func(zz)
        '''
        
        prof_max_index = np.argmax(prof)
        prof_max = prof[prof_max_index]
        
        prof = prof - prof_max
        
        left = 0
        for x in range(prof_max_index, 0, -1):
            if prof[x] > self.FWHM_threshold and prof[x-1] < self.FWHM_threshold:
                if prof[x] - self.FWHM_threshold > self.FWHM_threshold - prof[x-1]:
                    left = x - 1
                else:
                    left = x
        
        right = 0
        # for x in range(prof_max_index, self.r*h-1):
        for x in range(prof_max_index, len(prof) - 1):
            if prof[x] > self.FWHM_threshold and prof[x+1] < self.FWHM_threshold:
                if prof[x] - self.FWHM_threshold > self.FWHM_threshold - prof[x+1]:
                    right = x + 1
                else:
                    right = x
        
        axial_FWHM = (right - left) * self.dz / self.r
        
        return axial_FWHM
    
    def generate_bmode(self, image_data):
        bmode = 20 * np.log10(image_data / np.max(image_data))
        bmode = np.clip(bmode, self.color_limit[0], self.color_limit[1])
        return bmode
    
    def norm_data(self, data, range_values):
        nr_max = range_values[1]
        nr_min = range_values[0]
        x_scaled = (data - np.min(data)) / (np.max(data) - np.min(data)) * (nr_max - nr_min) + nr_min
        
        return x_scaled
    

'''
ax = plt.subplot()


save_root = '/media/zhangzm/LinuxData/UltrasonicPrediction/predict_save/2021_08_01_23_40_09'
src_data_path = os.path.join(save_root, 'test_simulation_point_teacher.npy')
# src_image = cv2.imread(image_path)

# bbox_point = [44, 600, 53, 660] # x0, y0, x1, y1
# bbox = plt.Rectangle((bbox_point[0], bbox_point[1]), bbox_point[2] - bbox_point[0], bbox_point[3] - bbox_point[1], fill=False, edgecolor='red', linewidth=1)
# ax.add_patch(bbox)

# plt.imshow(src_image)
# plt.show()

test_data = np.load(src_data_path)
print(test_data.shape)

dx, dz= 0.1, 0.0246
gain = 50
r = 100
FWHM_threshold = 20 * np.log10(0.5)

metrics = Metrics(dx, dz, gain, r, FWHM_threshold)


lesion = [600, 44, 660, 53]
background = [600, 76, 660, 85]

print(metrics.get_Contrast_CNR(test_data, lesion, background))

FWHM_area = [500, 40, 700, 60]
print(metrics.get_lateral_FWHM(test_data, FWHM_area))
print(metrics.get_axial_FWHM(test_data, FWHM_area))
'''
