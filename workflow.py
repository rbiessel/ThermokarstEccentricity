import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgba2rgb
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table
from scipy import stats
from skimage import filters
import xarray as xr


# image = imread('./data/test.png')
# b5 = xr.open_dataarray('./data/prudhoe_B5.tif', engine='rasterio').squeeze()


files = ['canning_B8', 'prudhoe_B8_N', 'prudhoe_B8_S', 'kaktovik']

names = ['Canning', 'Prudhoe', 'Dalton', 'Kaktovik']

for i in range(len(files)):

    b5 = xr.open_dataarray(
        f'./data/{files[i]}.tif', engine='rasterio').squeeze()

    gray_image = b5.to_numpy()

    image = gray_image

    # plt.imshow(gray_image)
    # plt.show()

    val = filters.threshold_otsu(gray_image) / 2
    # print(gray_image.dtype)
    # print(gray_image.shape)
    # print(val)

    binary_image = gray_image < val

    thresh = binary_image

    square = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

    def multi_dil(im, num, element=square):
        for i in range(num):
            im = dilation(im, element)
        return im

    def multi_ero(im, num, element=square):
        for i in range(num):
            im = erosion(im, element)
        return im

    # binary_image = area_closing(binary_image, 50000)
    binary_image = multi_ero(binary_image, 1)
    binary_image = multi_dil(binary_image, 1)

    # binary_image = binary_image[1000:3000, 1000:3000]
    # image = image[1000:3000, 1000:3000]
    # binary_image = opening(binary_image)

    label_im = label(binary_image, connectivity=1)
    regions = regionprops(label_im)

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    istd = np.std(image) * 2
    immean = np.mean(image)

    ax[1].imshow(binary_image, cmap=plt.cm.gray)
    ax[0].imshow(image, cmap=plt.cm.gray,
                 vmin=immean - istd, vmax=immean + istd)

    ax[0].set_title('a)', loc='left')
    ax[1].set_title('b)', loc='left')

    area_thresh = 10
    area_max = 100000

    for region in regions:
        if region.area > area_thresh and region.area <= area_max:
            y0, x0 = region.centroid
            orientation = region.orientation
            x1 = x0 + np.cos(orientation) * 0.5 * region.axis_minor_length
            y1 = y0 - np.sin(orientation) * 0.5 * region.axis_minor_length
            x2 = x0 - np.sin(orientation) * 0.5 * region.axis_major_length
            y2 = y0 - np.cos(orientation) * 0.5 * region.axis_major_length

            ax[0].plot((x0, x1), (y0, y1), '-r', linewidth=1)
            ax[0].plot((x0, x2), (y0, y2), '-r', linewidth=1)
            ax[0].plot(x0, y0, '.', color='red', markersize=5)
            ax[0].set_xticks([])
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[0].set_yticks([])

            # minr, minc, maxr, maxc = region.bbox
            # bx = (minc, maxc, maxc, minc, minc)
            # by = (minr, minr, maxr, maxr, minr)
            # ax[0].plot(bx, by, '-b', linewidth=1)

    plt.tight_layout()
    plt.show()

    properties = ['centroid', 'area', 'convex_area', 'extent', 'eccentricity',
                  'orientation']
    df = pd.DataFrame(regionprops_table(label_im, binary_image,
                                        properties=properties))

    # df.hist('eccentricity')
    # print(df['area'])

    lakes = df.loc[df['area'] > area_thresh]
    lakes = lakes.loc[lakes['area'] <= area_max]
    lakes['orientation'] = lakes['orientation'] - np.pi/2

    lakes.to_csv(
        f'/Users/rbiessel/Documents/DIP_Project/output/{names[i]}.csv')

    eccentricies = lakes['eccentricity']
    # kernel = stats.gaussian_kde(eccentricies)
    # x_pts = np.linspace(0, 1, 100)
    # estimated_pdf = kernel.evaluate(x_pts)
    # estimated_pdf = estimated_pdf/np.sum(estimated_pdf) * 4
    # plt.plot(x_pts, estimated_pdf, color='orange')

    # plt.show()

    # plt.hist(lakes['eccentricity'], bins=100)
    # lakes['eccentricity'].plot(kind='density')
    # plt.show()

    # bins = plt.hist(lakes['orientation'], bins=100)
    # plt.show()

    # # plt.scatter(lakes['area'], lakes['eccentricity'])
    # plt.scatter(lakes['centroid-0'], lakes['orientation'], s=10)
    # plt.xlabel('X')
    # plt.show()

    # plt.scatter(lakes['centroid-1'], lakes['orientation'], s=10)
    # plt.xlabel('Y')
    # plt.show()

    # r = np.arange(0, 2, 0.01)
    # theta = 2 * np.pi * r

    bins = int(360/8)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # Add pi/4 to everything to correct it
    lakes['orientation'] = lakes['orientation'] + np.pi/4

    hist_kwargs = {
        'bins': bins,
        'weights': lakes['area'],
        'color': 'black',
        'alpha': 0.7,
        'density': False
    }

    ax.hist(lakes['orientation'], **hist_kwargs)
    ax.hist(np.pi + lakes['orientation'], **hist_kwargs)
    ax.set_xticklabels(['N', 'NW', 'W', 'SW', 'S', 'SE', 'E', 'NE'])

    # ax.plot(lakes['orientation'], n)
    # ax.set_rmax(10)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    # ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    plt.savefig(
        f'/Users/rbiessel/Documents/DIP_Project/figures/{names[i]}.rose.png', transparent=True, dpi=300)
