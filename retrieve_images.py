import sys
import os
import time
import numpy as np
import cv2
import pickle
import datetime
from safebeach_detector import *
import gc
from os import listdir
from shutil import copyfile


def retrieve_process(beachcam="https://video-auth1.iol.pt/beachcam/conceicao/playlist.m3u8",cfg=PredictionConfig()):

    f = open("clf.pkl", "rb")
    clf = pickle.load(f)
    f.close()

    # define the model
    print("Building model...")
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    print("Loading weights...")
    if os.path.isdir('/content/'):
        model.load_weights("/content/mask_rcnn_beachsafe_cfg_0012.h5", by_name=True)
    else:
        model.load_weights("mask_rcnn_beachsafe_cfg_0012.h5", by_name=True)

    n_wanted = 10
    values = []
    occupations = []

    # curr_date = datetime.datetime.now()
    # curr_date = curr_date.strftime("%H-%M-%S")
    # os.mkdir('./%s/%s' %
    #          (folder, curr_date))

    num_max_frames = 100
    moving_threshold = 0.3  # Lower number means stroger filter

    stream = cv2.VideoCapture(beachcam)

    r, f = stream.read()
    if not r:
        return

    h, w, c = f.shape

    bord = int(w/40)
    borderL_prev = f[1:-1, 2:bord, :]
    borderR_prev = f[1:-1, w-bord:w-2, :]
    ind0 = 0
    ind1 = 0
    ind3 = 0
    accum_frame = np.zeros([num_max_frames, h-100, w, c])

    dif_borderL = np.array((0, borderL_prev.mean()))
    dif_borderR = np.array((0, borderR_prev.mean()))

    while ind3 < n_wanted:
        r, f = stream.read()
        # cv2.imshow('IP Camera stream', f)
        if r == True:
            # Check borders to detect if camera is static
            borderL = f[1:-1, 2:bord, :]
            borderR = f[1:-1, w-bord:w-2, :]

            dif_borderL = np.append(
                dif_borderL, [borderL_prev.mean() - borderL.mean()], axis=0)
            dif_borderR = np.append(
                dif_borderR, [borderR_prev.mean() - borderR.mean()], axis=0)
            borderL_prev = borderL
            borderR_prev = borderR

            if ind0 > 10:
                dif_borderL = dif_borderL[-10:]
                mean_difL = dif_borderL.mean()
                dif_borderR = dif_borderR[-10:]
                mean_difR = dif_borderR.mean()

                if abs(mean_difL) + abs(mean_difR) > moving_threshold:
                    # print(abs(mean_difL), abs(mean_difR), 'moving')
                    accum_frame = np.zeros([num_max_frames, h-100, w, c])
                    ind1 = 0
                    # cv2.imshow('IP Camera stream', f)
                    # accum_frame = np.append(accum_frame, [f], axis=0)
                elif abs(mean_difL) + abs(mean_difR) <= moving_threshold:
                    # print(abs(mean_difL), abs(mean_difR), 'not moving')
                    # Accumulate frames
                    accum_frame[ind1, :, :, :] = f[100:h, :, :]
                    ind1 = ind1 + 1

                    # Calculate median and send to thread to process
                    if ind1 == num_max_frames:
                        center = accum_frame[int(ind1 / 2), :, :, :]
                        median = np.median(accum_frame, axis=0)
                        
                        #print(center)
                        img_float32 = np.float32(center)
                        median_float32 = np.float32(median)
                        img = cv2.cvtColor(img_float32, cv2.COLOR_BGR2RGB)
                        median_cv = median_float32


                        
                        # convert pixel values (e.g. center)
                        scaled_image = mold_image(img, cfg)
                        # convert image into one sample
                        sample = expand_dims(scaled_image, 0)
                        # make prediction
                        yhat = model.detect(sample, verbose=0)

                        print("Persons found: ",len(yhat[0]['rois']))

                        mask = masked_image(median_cv,clf)
                        #plot_img_mask(img,mask)
                        occupation = calculate_occupation(mask,yhat[0]['rois'])
                        print("Occupation:", occupation)
                        #################################################################################
                        # POR AQUI FUNÇÃO PARA RETORNAR NUMERO DE PESSOAS E FAZER value.append(pessoas) #
                        # center é o frame do meio, median a mediana para a mascara,                    #
                        # podes enviar o modelo clf assim vai menos ao ficheiro buscar                  #
                        #################################################################################
                        values.append(len(yhat[0]['rois']))
                        occupations.append(occupation)

                        del img_float32,img,scaled_image,sample,yhat,mask,occupation,median_cv,median_float32,center,median
                        gc.collect()
                        ind3 = ind3 + 1

                        accum_frame = np.zeros([num_max_frames, h-100, w, c])
                        ind1 = 0

            ind0 = ind0+1
        elif r == False:
            time.sleep(0.01)
            stream = cv2.VideoCapture(beachcam)
            r, f = stream.read()
    stream.release()
    del model
    return values, occupations