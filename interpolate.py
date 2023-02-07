import numpy as np
from scipy.interpolate import RegularGridInterpolator

# def linear_interpolate(original_array, target_shape):
#     ts0, ts1 = target_shape
#     ogarr = np.array(original_array)
#     newarr = np.zeros(target_shape)
#     ri, rj = ogarr.shape[0]/target_shape[0], ogarr.shape[1]/target_shape[1]
#     x, y = np.arange(ogarr.shape[0]), np.arange(ogarr.shape[1])
#     grid = RegularGridInterpolator((x,y), ogarr)
#     points = np.zeros([ts0,ts1,2])
#     for i in range(ts0):
#         for j in range(ts1):
#             points[i,j] = [i*ri, j*rj]
#     newarr = grid(points)
#     return newarr

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

def mask_interpolate(original_array, target_shape):
    ts0, ts1 = target_shape
    ogarr = np.array(original_array)
    os0, os1 = ogarr.shape
    if target_shape == ogarr.shape:
        return ogarr
    x, y = np.arange(os0), np.arange(os1)
    grid = RegularGridInterpolator((x,y), ogarr)
    grid2 = RegularGridInterpolator((x,y), ogarr,method='nearest')
    a, b = np.meshgrid(np.linspace(0,os0-1,ts0), np.linspace(0,os1-1,ts1),indexing='ij')
    points = np.concatenate((a.reshape(ts0,ts1,1),b.reshape(ts0,ts1,1)),axis=-1)
    newarr = grid(points)
    newarr2 = grid2(points)
    outarr = newarr[:]
    for i, row in enumerate(points):
        for j, point in enumerate(row):
            if newarr[i,j] != newarr2[i,j]:
                p0,p1 = point
                newpts = np.array([[int(p0),int(p1)],[int(p0),int(np.ceil(p1))],
                                [int(np.ceil(p0)),int(p1)],[int(np.ceil(p0)),int(np.ceil(p1))]])
                vals = grid(newpts)
                outarr[i,j] = vals[np.argmin(abs(newarr[i,j]-vals))]
    return outarr

def extend_background(original_array,target_shape,background_array=None):
    os0, os1 = original_array.shape
    ts0, ts1 = target_shape
    if background_array is None:
        out = np.median(original_array[-1])*np.ones([ts0,ts1])
    else:
        out = np.mean(original_array[background_array>0]) + np.std(original_array[background_array>0])*np.random.randn(ts0,ts1)/10
    edge = (ts1-os1)//2
    out[-os0:,edge:-edge] = original_array
    return out


