import tifffile
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import random
from glob import glob
from natsort import natsorted
from interpolate import linear_interpolate
from cellpose_omni import models, core
from cellpose_omni import plot
from error_algorithms import IoU
from error_algorithms import centroid_distances
from error_algorithms import ignore_duplicates
from piqa import SSIM, PSNR
from tqdm import tqdm

def open_image(filename):
    if "tif" in filename:
        img = np.array(tifffile.imread(filename),dtype=np.float32)
    elif "png" in filename:
        img = np.array(Image.open(filename),dtype=np.float32)
    else:
        raise TypeError(f"Images must be tif files or png files, not {filename[-3:]}")
    return img

def do_everything_before(SR_TRAINING_IMAGES_DIR, HR_FOLDER_NAME, LR_FOLDER_NAME,METHOD="linear"):
    methods = ['linear','nearest','slinear','cubic','quintic','pchip']
    if METHOD not in methods:
        print(f"{METHOD} is not a valid method. Valid methods are: {methods}")
        raise TypeError
    HR_TRAIN_DIR = SR_TRAINING_IMAGES_DIR + "/train/" + HR_FOLDER_NAME + "/convolutions/"
    HR_TEST_DIR  = SR_TRAINING_IMAGES_DIR + "/test/"  + HR_FOLDER_NAME + "/convolutions/"
    LR_TRAIN_DIR = SR_TRAINING_IMAGES_DIR + "/train/" + LR_FOLDER_NAME + "/convolutions/"
    LR_TEST_DIR  = SR_TRAINING_IMAGES_DIR + "/test/"  + LR_FOLDER_NAME + "/convolutions/"
    hr_image = open_image(HR_TRAIN_DIR+os.listdir(HR_TRAIN_DIR)[0])
    UPSAMPLED_TRAIN_DIR = SR_TRAINING_IMAGES_DIR + "/train/" + LR_FOLDER_NAME + "_" + METHOD + "/"
    UPSAMPLED_TEST_DIR  = SR_TRAINING_IMAGES_DIR + "/test/"  + LR_FOLDER_NAME + "_" + METHOD + "/"
    SAVE_OUTPUT_DIR     = SR_TRAINING_IMAGES_DIR + "/train/" + LR_FOLDER_NAME + "_" + METHOD + "_output/"
    try:
        os.mkdir(UPSAMPLED_TRAIN_DIR)
    except FileExistsError:
        print(f"folder already exists: {UPSAMPLED_TRAIN_DIR}")
    try:
        os.mkdir(UPSAMPLED_TEST_DIR)
    except FileExistsError:
        print(f"folder already exists: {UPSAMPLED_TEST_DIR}")
    for file in os.listdir(LR_TRAIN_DIR):
        lr_image = open_image(LR_TRAIN_DIR+file)
        upsampled = np.rint(linear_interpolate(lr_image,target_shape=hr_image.shape,method=METHOD))//256
        upsampled = Image.fromarray(upsampled).convert("L")
        upsampled.save(UPSAMPLED_TRAIN_DIR+file[:-3]+"png")
    for file in os.listdir(LR_TEST_DIR):
        lr_image = open_image(LR_TEST_DIR+file)
        upsampled = np.rint(linear_interpolate(lr_image,target_shape=hr_image.shape,method=METHOD))//256
        upsampled = Image.fromarray(upsampled).convert("L")
        upsampled.save(UPSAMPLED_TEST_DIR+file[:-3]+"png")
    return HR_TRAIN_DIR, HR_TEST_DIR, UPSAMPLED_TRAIN_DIR, UPSAMPLED_TEST_DIR, SAVE_OUTPUT_DIR

def get_train_command(NUM_EPOCHS, HR_TRAIN_DIR, HR_TEST_DIR, UPSAMPLED_TRAIN_DIR, UPSAMPLED_TEST_DIR, SAVE_OUTPUT_DIR):
    command = f'python train.py --epochs {NUM_EPOCHS} --hr-path "{HR_TRAIN_DIR}" --lr-path "{UPSAMPLED_TRAIN_DIR}" --hr-validation-path "{HR_TEST_DIR}" --lr-validation-path "{UPSAMPLED_TEST_DIR}" --save-path "{SAVE_OUTPUT_DIR}"'
    return command

def do_everything_between(SEGMENTATION_IMAGES_DIR, HR_FOLDER_NAME, LR_FOLDER_NAME, SR_MODEL=None,METHOD="linear",HR_ONLY=False,TILE_LENGTH=40,TRAINING_SAMPLES=200,SR_PREFIX="SR",device=torch.device("cuda")):
    if  HR_ONLY:
        HR_TRAIN_DIR_MASKS = SEGMENTATION_IMAGES_DIR + "/train/" + HR_FOLDER_NAME + "/masks/"
        HR_TEST_DIR_MASKS  = SEGMENTATION_IMAGES_DIR + "/test/"  + HR_FOLDER_NAME + "/masks/"
        HR_TRAIN_DIR_CONVS = SEGMENTATION_IMAGES_DIR + "/train/" + HR_FOLDER_NAME + "/convolutions/"
        HR_TEST_DIR_CONVS  = SEGMENTATION_IMAGES_DIR + "/test/"  + HR_FOLDER_NAME + "/convolutions/"
        HR_TILED_TRAIN_DIR = SEGMENTATION_IMAGES_DIR + "/train/" + HR_FOLDER_NAME + f"_tiled/"
        HR_TILED_TEST_DIR  = SEGMENTATION_IMAGES_DIR + "/test/"  + HR_FOLDER_NAME + f"_tiled/"
        HR_TILED_TEST_SEG_DIR  = SEGMENTATION_IMAGES_DIR + "/test/"  + HR_FOLDER_NAME + f"_tiled_segmentations/"

        try:
            os.mkdir(HR_TILED_TRAIN_DIR)
        except FileExistsError:
            print(f"folder already exists: {HR_TILED_TRAIN_DIR}")
        try:
            os.mkdir(HR_TILED_TEST_DIR)
        except FileExistsError:
            print(f"folder already exists: {HR_TILED_TEST_DIR}")
        
        MASKS_TRAIN = sorted(glob(HR_TRAIN_DIR_MASKS+"/*"))
        MASKS_TEST  = sorted(glob(HR_TEST_DIR_MASKS+"/*"))
        CONVS_TRAIN = sorted(glob(HR_TRAIN_DIR_CONVS+"/*"))
        CONVS_TEST  = sorted(glob(HR_TEST_DIR_CONVS+"/*"))

        TRAIN_INDICES = random.sample(range(len(MASKS_TRAIN)-TILE_LENGTH), TRAINING_SAMPLES)
        TEST_SAMPLES = len(MASKS_TEST)//TILE_LENGTH
        TEST_INDICES = np.linspace(0,TILE_LENGTH*(TEST_SAMPLES-1),TEST_SAMPLES).astype(int)

        try:
            os.mkdir(HR_TILED_TEST_SEG_DIR)
        except FileExistsError:
            print(f"folder already exists: {HR_TILED_TEST_SEG_DIR}")
        
        for i, x in enumerate(TRAIN_INDICES):
            x = TRAIN_INDICES[i]
            mask_tile = np.concatenate([np.array(Image.open(mask)) for mask in MASKS_TRAIN[x:x+TILE_LENGTH]], axis=1)
            conv_tile = np.concatenate([np.array(Image.open(conv)) for conv in CONVS_TRAIN[x:x+TILE_LENGTH]], axis=1)
            Image.fromarray(mask_tile).save(f"{HR_TILED_TRAIN_DIR}/train_{str(i).zfill(5)}_masks.png")
            Image.fromarray(conv_tile).save(f"{HR_TILED_TRAIN_DIR}/train_{str(i).zfill(5)}.png")
        for i, x in enumerate(TEST_INDICES):
            x = TEST_INDICES[i]
            mask_tile = np.concatenate([np.array(Image.open(mask)) for mask in MASKS_TEST[x:x+TILE_LENGTH]], axis=1)
            conv_tile = np.concatenate([np.array(Image.open(conv)) for conv in CONVS_TEST[x:x+TILE_LENGTH]], axis=1)
            Image.fromarray(mask_tile).save(f"{HR_TILED_TEST_DIR}/test_{str(i).zfill(5)}_masks.png")
            Image.fromarray(conv_tile).save(f"{HR_TILED_TEST_DIR}/test_{str(i).zfill(5)}.png")
        return HR_TILED_TRAIN_DIR, HR_TILED_TEST_DIR, HR_TILED_TEST_SEG_DIR
    elif SR_MODEL is None and not HR_ONLY:
        methods = ['linear','nearest','slinear','cubic','quintic','pchip']
        if METHOD not in methods:
            print(f"{METHOD} is not a valid method. Valid methods are: {methods}")
            raise TypeError
        HR_TRAIN_DIR = SEGMENTATION_IMAGES_DIR + "/train/" + HR_FOLDER_NAME + "/masks/"
        HR_TEST_DIR  = SEGMENTATION_IMAGES_DIR + "/test/"  + HR_FOLDER_NAME + "/masks/"
        LR_TRAIN_DIR = SEGMENTATION_IMAGES_DIR + "/train/" + LR_FOLDER_NAME + "/convolutions/"
        LR_TEST_DIR  = SEGMENTATION_IMAGES_DIR + "/test/"  + LR_FOLDER_NAME + "/convolutions/"
        if len(os.listdir(HR_TRAIN_DIR)) != len(os.listdir(LR_TRAIN_DIR)):
            print(f"HR and LR training folders should be the same length, but have lengths {len(os.listdir(HR_TRAIN_DIR))}, {len(os.listdir(LR_TRAIN_DIR))}")
            raise ValueError
        if len(os.listdir(HR_TEST_DIR)) != len(os.listdir(LR_TEST_DIR)):
            print(f"HR and LR training folders should be the same length, but have lengths {len(os.listdir(HR_TRAIN_DIR))}, {len(os.listdir(LR_TRAIN_DIR))}")
            raise ValueError
        hr_image = open_image(HR_TRAIN_DIR+os.listdir(HR_TRAIN_DIR)[0])
        UPSAMPLED_TRAIN_DIR       = SEGMENTATION_IMAGES_DIR + "/train/" + LR_FOLDER_NAME + f"_{METHOD}/"
        UPSAMPLED_TEST_DIR        = SEGMENTATION_IMAGES_DIR + "/test/"  + LR_FOLDER_NAME + f"_{METHOD}/"
        UPSAMPLED_TILED_TRAIN_DIR = SEGMENTATION_IMAGES_DIR + "/train/" + LR_FOLDER_NAME + f"_{METHOD}_tiled/"
        UPSAMPLED_TILED_TEST_DIR  = SEGMENTATION_IMAGES_DIR + "/test/"  + LR_FOLDER_NAME + f"_{METHOD}_tiled/"
        UPSAMPLED_TILED_TEST_SEG_DIR  = SEGMENTATION_IMAGES_DIR + "/test/"  + LR_FOLDER_NAME + f"_{METHOD}_tiled_segmentations/"

        try:
            os.mkdir(UPSAMPLED_TRAIN_DIR)
        except FileExistsError:
            print(f"folder already exists: {UPSAMPLED_TRAIN_DIR}")
        try:
            os.mkdir(UPSAMPLED_TEST_DIR)
        except FileExistsError:
            print(f"folder already exists: {UPSAMPLED_TEST_DIR}")
        try:
            os.mkdir(UPSAMPLED_TILED_TRAIN_DIR)
        except FileExistsError:
            print(f"folder already exists: {UPSAMPLED_TILED_TRAIN_DIR}")
        try:
            os.mkdir(UPSAMPLED_TILED_TEST_DIR)
        except FileExistsError:
            print(f"folder already exists: {UPSAMPLED_TILED_TEST_DIR}")

        utrd = os.listdir(UPSAMPLED_TRAIN_DIR)
        uted = os.listdir(UPSAMPLED_TEST_DIR)

        for file in os.listdir(LR_TRAIN_DIR):
            if file[:-3]+"png" in utrd:
                continue
            lr_image = open_image(LR_TRAIN_DIR+file)
            upsampled = linear_interpolate(lr_image,target_shape=hr_image.shape,method=METHOD)
            upsampled = upsampled//256
            upsampled = Image.fromarray(upsampled).convert("L")
            upsampled.save(UPSAMPLED_TRAIN_DIR+file[:-3]+"png")
        for file in os.listdir(LR_TEST_DIR):
            if file[:-3]+"png" in uted:
                continue
            lr_image = open_image(LR_TEST_DIR+file)
            upsampled = linear_interpolate(lr_image,target_shape=hr_image.shape,method=METHOD)
            upsampled = upsampled//256
            upsampled = Image.fromarray(upsampled).convert("L")
            upsampled.save(UPSAMPLED_TEST_DIR+file[:-3]+"png")
        
        MASKS_TRAIN = sorted(glob(HR_TRAIN_DIR+"/*"))
        MASKS_TEST  = sorted(glob(HR_TEST_DIR+"/*"))
        CONVS_TRAIN = sorted(glob(UPSAMPLED_TRAIN_DIR+"/*"))
        CONVS_TEST  = sorted(glob(UPSAMPLED_TEST_DIR+"/*"))

        TRAIN_INDICES = random.sample(range(len(MASKS_TRAIN)-TILE_LENGTH), TRAINING_SAMPLES)
        TEST_SAMPLES = len(MASKS_TEST)//TILE_LENGTH
        TEST_INDICES = np.linspace(0,TILE_LENGTH*(TEST_SAMPLES-1),TEST_SAMPLES).astype(int)

        try:
            os.mkdir(UPSAMPLED_TILED_TEST_SEG_DIR)
        except FileExistsError:
            print(f"folder already exists: {UPSAMPLED_TILED_TEST_SEG_DIR}")
        
        for i, x in enumerate(TRAIN_INDICES):
            x = TRAIN_INDICES[i]
            mask_tile = np.concatenate([np.array(Image.open(mask)) for mask in MASKS_TRAIN[x:x+TILE_LENGTH]], axis=1)
            conv_tile = np.concatenate([np.array(Image.open(conv)) for conv in CONVS_TRAIN[x:x+TILE_LENGTH]], axis=1)
            Image.fromarray(mask_tile).save(f"{UPSAMPLED_TILED_TRAIN_DIR}/train_{str(i).zfill(5)}_masks.png")
            Image.fromarray(conv_tile).save(f"{UPSAMPLED_TILED_TRAIN_DIR}/train_{str(i).zfill(5)}.png")
        for i, x in enumerate(TEST_INDICES):
            x = TEST_INDICES[i]
            mask_tile = np.concatenate([np.array(Image.open(mask)) for mask in MASKS_TEST[x:x+TILE_LENGTH]], axis=1)
            conv_tile = np.concatenate([np.array(Image.open(conv)) for conv in CONVS_TEST[x:x+TILE_LENGTH]], axis=1)
            Image.fromarray(mask_tile).save(f"{UPSAMPLED_TILED_TEST_DIR}/test_{str(i).zfill(5)}_masks.png")
            Image.fromarray(conv_tile).save(f"{UPSAMPLED_TILED_TEST_DIR}/test_{str(i).zfill(5)}.png")
        return UPSAMPLED_TILED_TRAIN_DIR, UPSAMPLED_TILED_TEST_DIR, UPSAMPLED_TILED_TEST_SEG_DIR
    else:
        methods = ['linear','nearest','slinear','cubic','quintic','pchip']
        if METHOD not in methods:
            print(f"{METHOD} is not a valid method. Valid methods are: {methods}")
            raise TypeError
        HR_TRAIN_DIR = SEGMENTATION_IMAGES_DIR + "/train/" + HR_FOLDER_NAME + "/masks/"
        HR_TEST_DIR  = SEGMENTATION_IMAGES_DIR + "/test/"  + HR_FOLDER_NAME + "/masks/"
        LR_TRAIN_DIR = SEGMENTATION_IMAGES_DIR + "/train/" + LR_FOLDER_NAME + "/convolutions/"
        LR_TEST_DIR  = SEGMENTATION_IMAGES_DIR + "/test/"  + LR_FOLDER_NAME + "/convolutions/"
        if len(os.listdir(HR_TRAIN_DIR)) != len(os.listdir(LR_TRAIN_DIR)):
            print(f"HR and LR training folders should be the same length, but have lengths {len(os.listdir(HR_TRAIN_DIR))}, {len(os.listdir(LR_TRAIN_DIR))}")
            raise ValueError
        if len(os.listdir(HR_TEST_DIR)) != len(os.listdir(LR_TEST_DIR)):
            print(f"HR and LR training folders should be the same length, but have lengths {len(os.listdir(HR_TRAIN_DIR))}, {len(os.listdir(LR_TRAIN_DIR))}")
            raise ValueError
        hr_image = open_image(HR_TRAIN_DIR+os.listdir(HR_TRAIN_DIR)[0])
        SR_TRAIN_DIR       = SEGMENTATION_IMAGES_DIR + "/train/" + LR_FOLDER_NAME + f"_{METHOD}_{SR_PREFIX}/"
        SR_TEST_DIR        = SEGMENTATION_IMAGES_DIR + "/test/"  + LR_FOLDER_NAME + f"_{METHOD}_{SR_PREFIX}/"
        SR_TILED_TRAIN_DIR = SEGMENTATION_IMAGES_DIR + "/train/" + LR_FOLDER_NAME + f"_{METHOD}_{SR_PREFIX}_tiled/"
        SR_TILED_TEST_DIR  = SEGMENTATION_IMAGES_DIR + "/test/"  + LR_FOLDER_NAME + f"_{METHOD}_{SR_PREFIX}_tiled/"
        SR_TILED_TEST_SEG_DIR  = SEGMENTATION_IMAGES_DIR + "/test/"  + LR_FOLDER_NAME + f"_{METHOD}_{SR_PREFIX}_tiled_segmentations/"

        try:
            os.mkdir(SR_TRAIN_DIR)
        except FileExistsError:
            print(f"folder already exists: {SR_TRAIN_DIR}")
        try:
            os.mkdir(SR_TEST_DIR)
        except FileExistsError:
            print(f"folder already exists: {SR_TEST_DIR}")
        try:
            os.mkdir(SR_TILED_TRAIN_DIR)
        except FileExistsError:
            print(f"folder already exists: {SR_TILED_TRAIN_DIR}")
        try:
            os.mkdir(SR_TILED_TEST_DIR)
        except FileExistsError:
            print(f"folder already exists: {SR_TILED_TEST_DIR}")

        for file in tqdm(os.listdir(LR_TRAIN_DIR)):
            lr_image = open_image(LR_TRAIN_DIR+file)
            upsampled = linear_interpolate(lr_image,target_shape=hr_image.shape,method=METHOD)
            upsampled = upsampled.reshape(1,1,hr_image.shape[0],hr_image.shape[1])
            upsampled = (upsampled//256) #/255
            upsampled = torch.tensor(upsampled, dtype=torch.float)
            upsampled = upsampled.to(device)
            # print(f"the device here is {upsampled.device}")
            with torch.no_grad():
                output = SR_MODEL(upsampled).cpu()
                sr_image = (np.array(output).reshape(hr_image.shape)).astype(int) # *255
            sr_image = Image.fromarray(sr_image).convert("L")
            sr_image.save(SR_TRAIN_DIR+file[:-3]+"png")
        for file in tqdm(os.listdir(LR_TEST_DIR)):
            lr_image = open_image(LR_TEST_DIR+file)
            upsampled = linear_interpolate(lr_image,target_shape=hr_image.shape,method=METHOD)
            upsampled = upsampled.reshape(1,1,hr_image.shape[0],hr_image.shape[1])
            upsampled = (upsampled//256) #/255
            upsampled = torch.tensor(upsampled, dtype=torch.float)
            upsampled = upsampled.to(device)
            # print(f"the device here is {upsampled.device}")
            with torch.no_grad():
                output = SR_MODEL(upsampled).cpu()
                sr_image = (np.array(output).reshape(hr_image.shape)).astype(int) # *255
            sr_image = Image.fromarray(sr_image).convert("L")
            sr_image.save(SR_TEST_DIR+file[:-3]+"png")
        
        MASKS_TRAIN = sorted(glob(HR_TRAIN_DIR+"/*"))
        MASKS_TEST  = sorted(glob(HR_TEST_DIR+"/*"))
        CONVS_TRAIN = sorted(glob(SR_TRAIN_DIR+"/*"))
        CONVS_TEST  = sorted(glob(SR_TEST_DIR+"/*"))

        TRAIN_INDICES = random.sample(range(len(MASKS_TRAIN)-TILE_LENGTH), TRAINING_SAMPLES)
        TEST_SAMPLES = len(MASKS_TEST)//TILE_LENGTH
        TEST_INDICES = np.linspace(0,TILE_LENGTH*(TEST_SAMPLES-1),TEST_SAMPLES).astype(int)

        try:
            os.mkdir(SR_TILED_TEST_SEG_DIR)
        except FileExistsError:
            print(f"folder already exists: {SR_TILED_TEST_SEG_DIR}")
        
        for i, x in enumerate(TRAIN_INDICES):
            x = TRAIN_INDICES[i]
            mask_tile = np.concatenate([np.array(Image.open(mask)) for mask in MASKS_TRAIN[x:x+TILE_LENGTH]], axis=1)
            conv_tile = np.concatenate([np.array(Image.open(conv)) for conv in CONVS_TRAIN[x:x+TILE_LENGTH]], axis=1)
            Image.fromarray(mask_tile).save(f"{SR_TILED_TRAIN_DIR}/train_{str(i).zfill(5)}_masks.png")
            Image.fromarray(conv_tile).save(f"{SR_TILED_TRAIN_DIR}/train_{str(i).zfill(5)}.png")
        for i, x in enumerate(TEST_INDICES):
            x = TEST_INDICES[i]
            mask_tile = np.concatenate([np.array(Image.open(mask)) for mask in MASKS_TEST[x:x+TILE_LENGTH]], axis=1)
            conv_tile = np.concatenate([np.array(Image.open(conv)) for conv in CONVS_TEST[x:x+TILE_LENGTH]], axis=1)
            Image.fromarray(mask_tile).save(f"{SR_TILED_TEST_DIR}/test_{str(i).zfill(5)}_masks.png")
            Image.fromarray(conv_tile).save(f"{SR_TILED_TEST_DIR}/test_{str(i).zfill(5)}.png")
        return SR_TILED_TRAIN_DIR, SR_TILED_TEST_DIR, SR_TILED_TEST_SEG_DIR

def do_everything_after(SR_TILED_TRAIN_DIR, SR_TILED_TEST_DIR, SR_TILED_TEST_SEG_DIR, TILE_LENGTH=40, model_number=-1):
    try:
        os.mkdir(SR_TILED_TEST_SEG_DIR)
    except FileExistsError:
        print(f"folder already exists: {SR_TILED_TEST_SEG_DIR}")
    all = sorted(glob(SR_TILED_TEST_DIR + "/*"))
    mask = [m for m in all if "mask" in m]
    conv = [c for c in all if "mask" not in c]
    print(f"length of masks: {len(mask)}, {len(conv)}")
    masks = [np.asarray(Image.open(file)) for file in mask]
    convs = [np.asarray(Image.open(file)) for file in conv]
    nimg = len(convs)
    model_list = natsorted(glob(SR_TILED_TRAIN_DIR+"models/*"))
    model_name = model_list[model_number]
    print(f"Using model: {model_name}")
    use_gpu = True
    model = models.CellposeModel(gpu=use_gpu,pretrained_model=model_name,omni=True,concatenation=True)
    chans = [0,0] #this means segment based on first channel, no second channel

    n = [0] # make a list of integers to select which images you want to segment
    n = range(nimg) # or just segment them all

    # define parameters
    mask_threshold = -1
    verbose = False # turn on if you want to see more output
    transparency = True # transparency in flow output
    rescale=None # give this a number if you need to upscale or downscale your images
    omni = True # we can turn off Omnipose mask reconstruction, not advised
    flow_threshold = 0. # default is .4, but only needed if there are spurious masks to clean up; slows down output
    resample = True #whether or not to run dynamics on rescaled grid or original grid
    segmentations, flows, styles = model.eval([convs[i] for i in n],channels=chans,rescale=rescale,mask_threshold=mask_threshold,transparency=transparency,
                                    flow_threshold=flow_threshold,omni=omni,resample=resample,verbose=verbose)
    for idx,i in enumerate(n):
        maski = segmentations[idx]
        im = Image.fromarray(maski)
        im.save(f"{SR_TILED_TEST_SEG_DIR}/omni_{str(idx).zfill(5)}.png")
    outlist = []
    excesslist = []
    nearlist = []
    distlist = []
    for i in n:
        mask = masks[i]
        seg = segmentations[i]
        # plt.imshow(mask,"gray")
        # plt.title(f"mask {i}")
        # plt.show()
        # plt.imshow(seg,"gray")
        # plt.title(f"seg {i}")
        # plt.show()
        width = mask.shape[1]//TILE_LENGTH
        for j in range(TILE_LENGTH):
            maskj = mask[:,width*j:width*(j+1)]
            segj = seg[:,width*j:width*(j+1)]
            out, excess = IoU(maskj,segj)
            outlist.append(out)
            excesslist.append(excess)
            nearest, dist = centroid_distances(maskj,segj)
            nearlist.append(nearest)
            distlist.append(dist)
    cells, dists, duplicates, indices = ignore_duplicates(nearlist, distlist)
    dists = np.append(dists, [np.max(dists)]*duplicates)
    perclist = []
    for out in outlist:
        for a,b in out:
            perclist.append(a/b)
    percarr = np.array(perclist)
    excessout = []
    for exc in excesslist:
        excessout.extend(exc)
    excessarr = np.array(excessout)
    return percarr, cells, dists, duplicates, indices, excessarr

def get_test_ssim_psnr(HR_IMAGES_DIR,SR_IMAGES_DIR):
    ssim_list = []
    psnr_list = []
    ssim = SSIM(n_channels=1) #.cuda()
    psnr = PSNR()
    for hr_img, sr_img in zip(os.listdir(HR_IMAGES_DIR),os.listdir(SR_IMAGES_DIR)):
        hr_img = open_image(HR_IMAGES_DIR + hr_img)
        sr_img = open_image(SR_IMAGES_DIR + sr_img)
        hr_img_torch = torch.tensor(hr_img.reshape(1,1,hr_img.shape[0],hr_img.shape[1]))/np.max(hr_img)
        sr_img_torch = torch.tensor(sr_img.reshape(1,1,sr_img.shape[0],sr_img.shape[1]))/np.max(sr_img)
        img_ssim = ssim(sr_img_torch,hr_img_torch)
        img_psnr = psnr(sr_img_torch,hr_img_torch)
        ssim_list.append(img_ssim)
        psnr_list.append(img_psnr)
    return ssim_list, psnr_list
