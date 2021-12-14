import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve
from scipy.ndimage import filters

SOBEL_0 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
SOBEL_45 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
SOBEL_90 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_135 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
SOBEL_LIST = [SOBEL_0, SOBEL_45, SOBEL_90, SOBEL_135]
SOBEL_TITLES = ['0 degree Sobel', '45 degree Sobel', '90 degree Sobel', '135 degree Sobel']

# https://stackoverflow.com/questions/45960192/using-numpy-as-strided-function-to-create-patches-tiles-rolling-or-sliding-w/45960193#45960193
def __window_nd__(a, window, steps = None, axis = None, gen_data = False):
        """
        Create a windowed view over `n`-dimensional input that uses an 
        `m`-dimensional window, with `m <= n`
        
        Parameters
        -------------
        a : Array-like
            The array to create the view on
            
        window : tuple or int
            If int, the size of the window in `axis`, or in all dimensions if 
            `axis == None`
            
            If tuple, the shape of the desired window.  `window.size` must be:
                equal to `len(axis)` if `axis != None`, else 
                equal to `len(a.shape)`, or 
                1
                
        steps : tuple, int or None
            The offset between consecutive windows in desired dimension
            If None, offset is one in all dimensions
            If int, the offset for all windows over `axis`
            If tuple, the steps along each `axis`.  
                `len(steps)` must me equal to `len(axis)`
    
        axis : tuple, int or None
            The axes over which to apply the window
            If None, apply over all dimensions
            if tuple or int, the dimensions over which to apply the window

        gen_data : boolean
            returns data needed for a generator
    
        Returns
        -------
        
        a_view : ndarray
            A windowed view on the input array `a`, or `a, wshp`, where `whsp` is the window shape needed for creating the generator
            
        """
        ashp = np.array(a.shape)
        
        if axis != None:
            axs = np.array(axis, ndmin = 1)
            assert np.all(np.in1d(axs, np.arange(ashp.size))), "Axes out of range"
        else:
            axs = np.arange(ashp.size)
            
        window = np.array(window, ndmin = 1)
        assert (window.size == axs.size) | (window.size == 1), "Window dims and axes don't match"
        wshp = ashp.copy()
        wshp[axs] = window
        assert np.all(wshp <= ashp), "Window is bigger than input array in axes"
        
        stp = np.ones_like(ashp)
        if steps:
            steps = np.array(steps, ndmin = 1)
            assert np.all(steps > 0), "Only positive steps allowed"
            assert (steps.size == axs.size) | (steps.size == 1), "Steps and axes don't match"
            stp[axs] = steps
    
        astr = np.array(a.strides)
        
        shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
        strides = tuple(astr * stp) + tuple(astr)
        
        as_strided = np.lib.stride_tricks.as_strided
        a_view = np.squeeze(as_strided(a, 
                                     shape = shape, 
                                     strides = strides))
        if gen_data :
            return a_view, shape[:-wshp.size]
        else:
            return a_view

# https://stackoverflow.com/questions/31774071/implementing-log-gabor-filter-bank
def __get_gabor_filter__(f_0, theta_0, N, n_orientations):
    # filter configuration
    scale_bandwidth =  0.996 * math.sqrt(2/3)
    angle_bandwidth =  0.996 * (1/math.sqrt(2)) * (np.pi/n_orientations)

    # x,y grid
    extent = np.arange(-N/2, N/2 + N%2)
    x, y = np.meshgrid(extent,extent)

    mid = int(N/2)
    ## orientation component ##
    theta = np.arctan2(y,x)
    center_angle = ((np.pi/n_orientations) * theta_0) if (f_0 % 2) \
                else ((np.pi/n_orientations) * (theta_0+0.5))

    # calculate (theta-center_theta), we calculate cos(theta-center_theta) 
    # and sin(theta-center_theta) then use atan to get the required value,
    # this way we can eliminate the angular distance wrap around problem
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    ds = sintheta * math.cos(center_angle) - costheta * math.sin(center_angle)    
    dc = costheta * math.cos(center_angle) + sintheta * math.sin(center_angle)  
    dtheta = np.arctan2(ds,dc)

    orientation_component =  np.exp(-0.5 * (dtheta/angle_bandwidth)**2)

    ## frequency componenet ##
    # go to polar space
    raw = np.sqrt(x**2+y**2)
    # set origin to 1 as in the log space zero is not defined
    raw[mid,mid] = 1
    # go to log space
    raw = np.log2(raw)

    center_scale = math.log2(N) - f_0
    draw = raw-center_scale
    frequency_component = np.exp(-0.5 * (draw/ scale_bandwidth)**2)

    # reset origin to zero (not needed as it is already 0?)
    frequency_component[mid,mid] = 0

    return frequency_component * orientation_component

def __edge_descriptor__(im, cell_size, sigma, gaussian_before):
    # For the paper equivalents
    # n = cell_size, k = stride, b = block size
    n = cell_size
    k = n
    b = n * 2

    # DM_k(x, y) = |fk(x, y) * I(x, y)|
    if gaussian_before:
        im = filters.gaussian_filter(im, sigma=sigma)

    DM_list = [abs(convolve(im, sobel_kernel)) for sobel_kernel in SOBEL_LIST]

    if not gaussian_before:
        DM_list = [filters.gaussian_filter(DM_k, sigma=sigma) for DM_k in DM_list]

    # DEBUGGING
    # for DM_k in DM_list:
    #     plt.figure()
    #     plt.imshow(DM_k, cmap='gray')

    # "We divide into p overlapping blocks and calculate the feature vector of each block"
    # "K is the stride between two neighbor blocks"
    # "The cell size is n x n"
    # "We set k = n in our experiments"
    DM_blocks = [__window_nd__(DM_k, (b, b), steps=k).reshape(-1, b, b) for DM_k in DM_list]

    # Get the center pixel
    c_row = b // 2
    c_col = b // 2

    # Get the offset to move the center from the middle of the block to the middle of one of the four cells
    row_offset = c_row // 2
    col_offset = c_col // 2

    # Get the four cell pixel centers
    centers = [
        (c_row - row_offset, c_col - col_offset),
        (c_row - row_offset, c_col + col_offset),
        (c_row + row_offset, c_col - col_offset),
        (c_row + row_offset, c_col + col_offset)
    ]

    feature_vector = []
    for current_block_idx in range(DM_blocks[0].shape[0]):
        block_vector = []
        for row_coord, col_coord in centers:
            for DM_block_idx in range(len(DM_blocks)):
                center_pixel = DM_blocks[DM_block_idx][current_block_idx, row_coord, col_coord]
                block_vector.append(center_pixel)
        feature_vector.extend(block_vector / np.linalg.norm(block_vector))
    
    assert len(feature_vector) == DM_blocks[0].shape[0] * 16, 'Shape mismatch on edge descriptor'

    return np.asarray(feature_vector), DM_list

def __texture_descriptor__(im, n_scales=4, n_orientations=6, N=16):
    All_Aos = []
    for o in range(n_orientations):
        Ao = 0
        for s in range(n_scales):
            # Get the 2d log-gabor filter to convolve
            g = __get_gabor_filter__(s, o, N=N, n_orientations=n_orientations)
            A = abs(convolve(im, g))
            # Add all the scales together as per the paper
            Ao += A
        All_Aos.append(Ao)
    
    # Combine into maximum index map (MIM)
    All_Aos = np.asarray(All_Aos)
    MIM = np.argmax(All_Aos, axis=0)

    # DEBUGGING
    # plt.figure()
    # plt.imshow(MIM, cmap='gray')

    # Divide into 6x6 sub-grids
    blocks = __window_nd__(MIM, (6, 6), steps=6).reshape(-1, 6, 6)

    feature_vector = []
    for block in blocks:
        hist, _ = np.histogram(block, n_orientations)
        feature_vector.append(hist)

    feature_vector = feature_vector / np.linalg.norm(feature_vector)

    return feature_vector.flatten(), MIM

def get_descriptor(im, cell_size, sigma, gaussian_before, n_scales, n_orientations, N, sampling_interval):
    edge_vector, _ = __edge_descriptor__(im, cell_size, sigma, gaussian_before)
    texture_vector, _ = __texture_descriptor__(im, n_scales, n_orientations, N)

    # print(type(edge_vector), type(texture_vector), len(edge_vector), len(texture_vector), edge_vector.shape, texture_vector.shape)
    return np.concatenate((edge_vector, texture_vector))[::sampling_interval]
    # return texture_vector[::sampling_interval]