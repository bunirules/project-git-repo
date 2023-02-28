import numpy as np
from scipy.interpolate import RegularGridInterpolator

def linear_interpolate(original_array, target_shape,method='linear'):
    ts0, ts1 = target_shape
    ogarr = np.array(original_array)
    os0, os1 = ogarr.shape
    if target_shape == ogarr.shape:
        return ogarr
    methods = ['linear','nearest','slinear','cubic','quintic','pchip']
    if method not in methods:
        print(f"{method} is not a valid method. Valid methods are: {methods}. \n Defaulting to linear interpolation.")
        method='linear'
    x, y = np.arange(os0), np.arange(os1)
    grid = RegularGridInterpolator((x,y), ogarr,method=method)
    a, b = np.meshgrid(np.linspace(0,int(os0-1),int(ts0)), np.linspace(0,int(os1-1),int(ts1)),indexing='ij')
    points = np.concatenate((a.reshape(int(ts0),int(ts1),1),b.reshape(int(ts0),int(ts1),1)),axis=-1)
    newarr = grid(points)
    return newarr

def extend_background(original_array,target_shape,background_array=None):
    os0, os1 = original_array.shape
    ts0, ts1 = target_shape
    if background_array is None:
        out = np.median(original_array[-1])*np.zeros([ts0,ts1])
    else:
        out = np.mean(original_array[background_array>0]) + np.std(original_array[background_array>0])*np.random.randn(ts0,ts1)/10
    edge = (ts1-os1)//2
    try:
        out[-os0:,edge:-edge] = original_array
    except:
        out[-os0:,edge:-edge-1] = original_array
    return out


