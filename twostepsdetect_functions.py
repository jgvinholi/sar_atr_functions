from matplotlib.image import imread
from matplotlib import rcParams
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy
import os, glob
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import fclusterdata
from numba import jit, njit
import warnings
from operator import itemgetter 
from joblib import Parallel, delayed
import multiprocessing
from focalloss import *
from basefunctions import *

# Definition of a simple class containing properties about speckle and gaussian noise.
class noiseStruct(object):
  def __init__(self, speckle_mean, speckle_var, gaussian_var, aug_negat_prob):
    self.speckle_mean = speckle_mean
    self.speckle_var = speckle_var
    self.gaussian_var = gaussian_var
    self.aug_negat_prob = aug_negat_prob


# This function adds speckle (multiplicative) and gaussian noise to a matrix image. Used to augment the database.
def add_speckle_noise_to_img(Image, speckle_mean, speckle_variance, gaussian_variance):
  gaussian_mean = 0
  Speckle_noise = np.random.normal(loc = speckle_mean, scale = np.sqrt(speckle_variance), size = Image.shape)
  Gaussian_noise = np.random.normal(loc = gaussian_mean, scale = np.sqrt(gaussian_variance), size = Image.shape)
  Image = np.floor(Image*Speckle_noise + Gaussian_noise)
  return Image



# This function cuts the specified image vector into multiple square windows to be used in training and validation. 
# Data augmentation with noise also takes place below.
def gen_classification_gt(Images_vector, img_name, window_size, overlap, noiseproperties_vec):
  print(img_name)
  effective_wsize = window_size - overlap
  npixels_vert = im_dims[0]
  npixels_horiz = im_dims[1]
  nblocks_vert = int( ( npixels_vert )/effective_wsize + 1)
  nblocks_horiz = int( ( npixels_horiz)/effective_wsize + 1)
  X_full_pixval_class_window = np.zeros( (nblocks_vert*nblocks_horiz, window_size, window_size) )
  Y_class_window = np.zeros( (nblocks_vert*nblocks_horiz, 1) )


  # Retrieves the ground truth of the specified image.
  tmp_mat = np.loadtxt(datab_imgs_path + "labels/" + img_name + ".txt", delimiter = ' ', usecols = range(4))
  tmp_mat = tmp_mat.astype(int)
  X_targets = tmp_mat[:, 0:2]
  Y_targets = tmp_mat[:, 2:4]
  X_targets = X_targets[ Y_targets[:, 0] == 1 , :]

  print("Windows to process: " + str(nblocks_vert*nblocks_horiz) )
  
  # Cutting windows with size = 'window_size' and overlap = 'overlap'. A padding of zeros is added to the border windows as needed.
  for wind_v, j in enumerate( range(0, npixels_vert, effective_wsize) ):
    if (j % 30*npixels_vert == 0): print("Processing window " + str(wind_v*nblocks_horiz) )
    for wind_h, k in enumerate( range(0, npixels_horiz, effective_wsize) ):
      if ( (k - overlap >= 0) and (j - overlap >= 0) and (k + effective_wsize <= im_dims[1]) and (j + effective_wsize <= im_dims[0]) ):
        vert_range = np.arange((j-overlap), (j+effective_wsize))
        horiz_range = np.arange((k-overlap), (k+effective_wsize))
        window_points = np.transpose([np.tile(vert_range, len(horiz_range)), np.repeat(horiz_range, len(vert_range))])
        X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, :, :] = np.reshape( Images_vector[window_points[:, 0], window_points[:, 1]], (window_size, window_size), order='F'  )
        target = np.any( [np.any( ( X_targets == window_points[i, :]).all( axis = 1 ) ) for i in range(window_points.shape[0] ) ] )
        Y_class_window[wind_h + wind_v*nblocks_horiz] = target
      else:
        for wpos_v, l in enumerate( range(-overlap, effective_wsize) ):
          for wpos_h, m in enumerate( range(-overlap, effective_wsize) ):
            if ( (k+m >= 0) and (j+l >= 0) and (k+m < im_dims[1]) and (j+l < im_dims[0]) ):
              X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, wpos_v, wpos_h] = Images_vector[ j + l, k + m]
              if np.any( ( X_targets == np.array([j + l, k + m]) ).all( axis = 1 ) ): 
                Y_class_window[wind_h + wind_v*nblocks_horiz] = 1
            else:
              X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, wpos_v, wpos_h] = 0


  addnoise = 1
  if addnoise:
    print("Adding multiplicative and additive noise radomly to nontarget windows. Augmenting database with noisy target windows.")

    # Adding noise to the negative windows with a rate of 'noiseproperties_vec[i_noisevec].aug_negat_prob'. Noisy positive windows are added to the dataset,
    # mantaining all non-noisy positive windows.
    npositives = np.sum(Y_class_window[:, 0] == 1)*len(noiseproperties_vec)
    X_noisypositives = np.zeros( ( npositives, X_full_pixval_class_window.shape[1], X_full_pixval_class_window.shape[2] ) )
    Y_noisypositives = np.ones( (npositives, 1) )
    j_pos = 0
    for i_block, block_class in enumerate(Y_class_window[:, 0]):
      for i_noisevec in range(len(noiseproperties_vec)):
        # print("Speckle variance = " + str(noiseproperties_vec[i_noisevec].speckle_var) + ". Gaussian variance = " + str(noiseproperties_vec[i_noisevec].gaussian_var) + "." )
        if int(block_class) == 0:
          # Generate Bernoulli RV in order to choose by chance the negative images to which noise will be added:
          addnegnoise = scipy.stats.bernoulli.rvs(noiseproperties_vec[i_noisevec].aug_negat_prob)
          if addnegnoise:
            X_full_pixval_class_window[i_block, :, :] = add_speckle_noise_to_img(X_full_pixval_class_window[i_block, :, :], 
              noiseproperties_vec[i_noisevec].speckle_mean, noiseproperties_vec[i_noisevec].speckle_var, noiseproperties_vec[i_noisevec].gaussian_var)
        
        elif int(block_class) == 1:
          X_noisypositives[j_pos, :, :] = add_speckle_noise_to_img(X_full_pixval_class_window[i_block, :, :], noiseproperties_vec[i_noisevec].speckle_mean,
            noiseproperties_vec[i_noisevec].speckle_var, noiseproperties_vec[i_noisevec].gaussian_var)
          j_pos += 1
    X_full_pixval_class_window = np.concatenate((X_full_pixval_class_window, X_noisypositives), axis = 0)
    Y_class_window = np.concatenate( (Y_class_window, Y_noisypositives), axis = 0 )
    
  # Standardizing all resulting windows.
  for i_window in range( X_full_pixval_class_window.shape[0] ):
    X_full_pixval_class_window[i_window, :, :] = standardize_pixels(X_full_pixval_class_window[i_window, :, :])
  
  return X_full_pixval_class_window, Y_class_window



# Executes the gen_classification_gt function for multiple images and saves all the generated windows in a file.
def gen_multiple_classification_gt(Images_vector, img_names, noiseprop_vec, save):
  n_images = len(img_names)
  effective_wsize = window_size - overlap
  npixels_vert = im_dims[0]
  npixels_horiz = im_dims[1]
  nblocks_vert = int( ( npixels_vert )/effective_wsize + 1)
  nblocks_horiz = int( ( npixels_horiz)/effective_wsize + 1)
  num_cores = multiprocessing.cpu_count()
  
  # Parallel processing all images.
  parallel_genclassgt = Parallel(n_jobs = num_cores, backend='threading')(delayed(gen_classification_gt)(Images_vector[i], img_names[i], window_size, overlap, noiseprop_vec) for i in range( n_images ) )
  X_full_pixval_class_window, Y_class_window = zip(*parallel_genclassgt)

  # Save all windows as single variable
  if save:
    with open(datab_imgs_path + 'classification_data/' + 'xy_classification_noaug.pkl' , 'wb') as f:  # Python 3: open(..., 'wb')
      pickle.dump([X_full_pixval_class_window, Y_class_window, noiseprop_vec], f)
      print('saved')

  return X_full_pixval_class_window, Y_class_window




# This function loads from disk the variables containing all windows created for classification training/validation. 
def load_multiple_classification_gt():
  # Load all windows that went through the noise data augmentation:
  with open(datab_imgs_path + 'classification_data/' + 'xy_classification.pkl' , 'rb') as f:  # Python 3: open(..., 'wb')
    try:
      X_full_pixval_class_window, Y_class_window, noiseprop_vec = pickle.load(f)
    except:
      X_full_pixval_class_window, Y_class_window = pickle.load(f)
    print('loaded')
  
  # Load all windows with no added noise:
  with open(datab_imgs_path + 'classification_data/' + 'xy_classification_noaug.pkl' , 'rb') as f:  # Python 3: open(..., 'wb')
    X_full_pixval_class_window_noaug, Y_class_window_noaug = pickle.load(f)
    print('loaded')
  return X_full_pixval_class_window, Y_class_window, X_full_pixval_class_window_noaug, Y_class_window_noaug




# This function generates the binary heatmap by evaluating the binary predetection model. 
# KFold index is determined by the position of the specified 'img_name' in the 'whole_set' list. 
def predict_image(model_conv, img_name, kfold, whole_set):
  Image_vec_norm = standardize_pixels( bwimg( os.path.join(datab_imgs_path, img_name + ".jpg") ) )
  Image_vec_norm = np.reshape(Image_vec_norm, (1, Image_vec_norm.shape[0], Image_vec_norm.shape[1], 1) )
  if kfold:
    n_split = len(whole_set)//4
    n_images = len(whole_set)
    kfold_img_index = int( (whole_set.index(img_name) // (n_images // n_split) ) % 5) 
    print(img_name, kfold_img_index)
    # Pass the image through the predetection model:
    Target_map = np.reshape( model_conv[kfold_img_index].predict(Image_vec_norm, verbose = 1), (3000, 2000) )
  else:
    Target_map = np.reshape( model_conv.predict(Image_vec_norm, verbose = 1), (3000, 2000) )
  return Target_map




# This function receives one cluster center of a potential target and applies the classification prediction model at the pixels near it.
# The cluster center will be the center coordinate of the window to be processed by the classification model.
def predict_class(model_classconv, img_name, trainval_dataset, cluster_center):
  n_split = len(trainval_dataset)//4
  n_images = len(trainval_dataset)
  try:
    image_index = trainval_dataset.index(img_name)
    kfold_index = int( ( image_index // (n_images // n_split) ) % 5)
  except ValueError:
    image_index = whole_set.index(img_name)
    kfold_index = 0
  
  
  Pred_window = np.zeros( (1, window_size, window_size, 1) )
  Picture = Images_vector[image_index]
  halfwindow = int( window_size/2 )

  unparity = 0 if halfwindow == window_size/2 else 1

  # Determine the limit coordinates of the window to be analized.
  lowbound_horiz = -halfwindow + cluster_center[1]
  highbound_horiz = halfwindow + cluster_center[1]
  
  lowbound_vert = -halfwindow + cluster_center[0]
  highbound_vert = halfwindow + cluster_center[0]

  # If the coordinates are not out of bound, a vectorized window extraction takes place. 
  if (lowbound_horiz >= 0) and (lowbound_vert >= 0) and (highbound_horiz < im_dims[1]) and (highbound_vert < im_dims[0]):
    Pred_window[0, :, :, 0] = Picture[lowbound_vert:highbound_vert, lowbound_horiz:highbound_horiz]
  else:
    # The windows that, at some point, exceed the image's bounds are padded with zeros:
    for i, i_vert in enumerate( range(-halfwindow, halfwindow + unparity) ):
      for j, j_horiz in enumerate( range(-halfwindow, halfwindow + unparity) ):
        if ( (i_vert + cluster_center[0]) >= 0 ) and ( (i_vert + cluster_center[0]) < im_dims[0] ) and ( (j_horiz + cluster_center[1]) >= 0) and ( (j_horiz + cluster_center[1]) < im_dims[1]) :
          Pred_window[0, i, j, 0] = Picture[ i_vert + cluster_center[0], j_horiz + cluster_center[1] ]
        else:
          Pred_window[0, i, j, 0] = 0

  # Execute the classification model over the extracted and standardized window:
  Pred_window = standardize_pixels(Pred_window)   
  prediction = model_classconv[kfold_index].predict(Pred_window, verbose = 0)

  return prediction





# Based on the predetection heatmap, this function localizes all detection clusters and calculates its centers by an weighted mean.
# Three choices of clustering techniques: morphological (advantages: none), DBSCAN (good noise suppression, best perfomance and smoother ROC), agglomerative (smooth ROC, but very noisy).
# DBSCAN is the technique of choice.
def find_clusters_andsave(model_classconv, Pred, detect_thresh, classif_thresh, img_name, trainval_dataset, save):
  clustering = "dbscan"
  n_split = len(trainval_dataset)//4
  n_images = len(trainval_dataset)
  try:
    image_index = trainval_dataset.index(img_name)
    kfold_index = int( image_index // (n_images // n_split) )
  except ValueError:
    image_index = whole_set.index(img_name)
    kfold_index = 0

  if (clustering == "dbscan"):
      Pred_mask = Pred > detect_thresh # Generate binary map.
      candidates_coordinates = np.transpose( np.nonzero(Pred_mask) )
      candidates_predval = Pred[candidates_coordinates[:, 0], candidates_coordinates[:, 1]] # Will be used as weights.
      candidates_predval /= np.mean(candidates_predval) # Normalize by mean.
      clusters = DBSCAN(eps = 5.01, min_samples = 8, n_jobs = -1 ).fit(candidates_coordinates) # Run DBSCAN. Maximum distance between pixels is of 5px. 8 samples are needed to form a cluster.
      unique_labels = np.array(list( set(clusters.labels_) ) ) # Convert to ndarray.
      unique_nonoise = unique_labels[unique_labels != -1] # Retrieve the number of found clusters
      
      possible_clusters = np.zeros( ( len(unique_nonoise), 2 ) , dtype = 'int32')
      possible_clusters_probs = np.zeros( ( len(unique_nonoise) ) )
      clusters_centers = []
      
      # Calculate found clusters' centers and feed to the classification model. For each cluster, if the result is bigger than a threshold, the cluster is assumed to be a target.
      j = 0
      for k in set(unique_nonoise):
        class_member_mask = (clusters.labels_ == k)
        class_members = candidates_coordinates[class_member_mask, :]
        possible_clusters[j, :] = np.average( class_members, axis = 0, weights = np.power(candidates_predval[class_member_mask], 2 ) ).astype('int32') # For each cluster, calculates the average position of all cluster members, weighted by the heatmap value of each member. 
        possible_clusters_probs[j] = np.reshape( predict_class(model_classconv, img_name, trainval_dataset, possible_clusters[j, :]), 1)
        if possible_clusters_probs[j] > classif_thresh:
          clusters_centers.append( possible_clusters[j, :]  ) # If the classificator returns a value bigger than 'classif_thresh', the cluster will be assumed to be a target.
        j += 1

      Final_mask = np.zeros( Pred.shape , dtype = np.uint8)
      if save:
        for i, cluster in enumerate( clusters_centers ):
          Final_mask = cv2.circle(Final_mask, (cluster[1], cluster[0]), radius = 7, color = 255, thickness = 2)
      
        # showimg(Final_mask)
        saveimg(Final_mask, os.path.join(datab_imgs_path, "predictions/" + img_name + "_dbscan_twosteps_prediction.jpg") )
        np.savetxt(os.path.join(datab_imgs_path, "predictions/" + img_name + "_cluster_centers_prediction.txt"), clusters_centers, "%d" )

  elif (clustering == "morphological"):
    Pred_mask = Pred > detect_thresh
    erode = 1
    dilation_kernel = np.ones((4, 4), np.uint8)
    erosion_kernel = np.ones((2, 2), np.uint8) # Erosion kernel is defined as 5x5 window
    if erode:
        img_erosion = cv2.erode(np.uint8(Pred_mask)*255, erosion_kernel, iterations=1) 
        img_dilation = cv2.dilate(img_erosion, dilation_kernel, iterations=1)
    else:
        img_dilation = cv2.dilate(np.uint8(Pred_mask)*255, dilation_kernel, iterations=2)
    img_contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contourned = np.zeros((img_dilation.shape[0], img_dilation.shape[1], 3))
    img_contourned[:, :, 0] = img_dilation
    img_contourned[:, :, 1] = img_dilation
    img_contourned[:, :, 2] = img_dilation
    cv2.drawContours(img_contourned, img_contours, -1, (127, 255, 0), 1)
    cluster_centers = np.zeros( (len(img_contours), 2) )
    for i in range( cluster_centers.shape[0] ):
        cluster_centers[i, :] = np.mean( img_contours[i], axis = 0 )
    reshaped_cluster_centers = np.copy(cluster_centers)
    reshaped_cluster_centers[:, 1] = reshaped_cluster_centers[:, 0]
    reshaped_cluster_centers[:, 0] = cluster_centers[:, 1]
    reshaped_cluster_centers = reshaped_cluster_centers.astype(int)
    if save:
        saveimg(img_contourned, os.path.join(datab_imgs_path, "predictions/" + img_name + "_morph_prediction.jpg") )
        np.savetxt(os.path.join(datab_imgs_path, "predictions/" + img_name + "_cluster_centers_prediction.txt"), reshaped_cluster_centers, "%d" )
    return img_contourned, reshaped_cluster_centers
   
  
  elif (clustering == "agglomerative"):
      Pred_mask = Pred > detect_thresh
      candidates_coordinates = np.transpose( np.nonzero(Pred_mask) )
      candidates_predval = Pred[candidates_coordinates[:, 0], candidates_coordinates[:, 1]] # Will be used as weights.
      candidates_predval /= np.mean(candidates_predval) # Normalize by mean.
      
      clusters = fclusterdata(candidates_coordinates, 15, criterion='distance')
      print(candidates_coordinates.shape, clusters.shape)
      unique_labels = np.array(list( set(clusters) ) ) # Convert to ndarray.
      print(unique_labels)
      unique_nonoise = unique_labels[unique_labels != -1]
      clusters_centers = np.zeros( ( len(unique_nonoise), 2 ) , dtype = 'int32')
      colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_nonoise))]
      j = 0
      for k, col in zip(set(unique_nonoise), colors):
        class_member_mask = (clusters == k)
        class_members = candidates_coordinates[class_member_mask, :]
        clusters_centers[j, :] = np.average( class_members, axis = 0, weights = candidates_predval[class_member_mask]).astype('int32')
        print(clusters_centers[j, :])
        j += 1

      Final_mask = np.zeros( Pred.shape , dtype = np.uint8)
      if save:
        for i in range( clusters_centers.shape[0] ):
          Final_mask = cv2.circle(Final_mask, (clusters_centers[i, 1], clusters_centers[i, 0]), radius = 6, color = 255, thickness = 2)

        showimg(Final_mask)
        saveimg(Final_mask, os.path.join(datab_imgs_path, "predictions/" + img_name + "_dbscan_prediction.jpg") )
        np.savetxt(os.path.join(datab_imgs_path, "predictions/" + img_name + "_cluster_centers_prediction.txt"), clusters_centers, "%d" )
  
  return Final_mask, clusters_centers







# This function recieves the predetection heatmap, calls the cluster localization function 'find_clusters_andsave' and calculates the detection performance indicators (true positive rate, false alarm rate, etc).
def detection_perfomance(model_conv, model_classconv, Pred, detect_thresh, classif_thresh, img_name, save):

  detected_targets = np.zeros((1, 2))
  undetected_targets = np.zeros((1, 2))
  correct_clusters_indices = np.zeros((1, 1))
  Img_GT_mask = np.zeros((3000, 2000))

  if (Pred == 0).all():
    Pred = predict_image(model_conv, img_name, 1, img_names) # Generates the predetection heatmap if none is provided.

  Pred_mask = Pred > detect_thresh # Apply detection threshold to generate the binary map.

  # Retreive clusters from predetection heatmap:
  _, cluster_centers = find_clusters_andsave(model_classconv, Pred, detect_thresh, classif_thresh, img_name, img_names, save)
  
  # Get ground truth position of target pixels:
  tmp_mat = np.loadtxt(datab_imgs_path + "labels/" + img_name + ".txt", delimiter = ' ', usecols = range(4))
  tmp_mat = tmp_mat.astype(int)
  X_targets = tmp_mat[:, 0:2]
  Y_targets = tmp_mat[:, 2:4]
  X_targets = X_targets[ Y_targets[:, 0] == 1 , :]
  Y_targets = Y_targets[ Y_targets[:, 0] == 1 , :]

  # For each found cluster (possible target), calculate the distance from all true targets based on the ground truth:
  for i in range(X_targets.shape[0]): 
    for j in range(len(cluster_centers ) ):
      # If a cluster is closer than 10px of a ground truth position, the cluster is declared to be a correct detection.
      distance_from_target = np.sqrt( (cluster_centers[j][0] - X_targets[i][0])**2 + (cluster_centers[j][1] - X_targets[i][1])**2 )
      if distance_from_target < 10: #10px = 10m
        correct_clusters_indices = np.vstack([correct_clusters_indices, j])
        detected_targets = np.vstack([detected_targets, X_targets[i, :] ] )
        break
              
  detected_targets = np.delete(detected_targets, 0, 0) # Delete first useless line.
  correct_clusters_indices = np.delete(correct_clusters_indices, 0, 0) # Delete first useless line.
  
  for i in range(X_targets.shape[0]): # Find undetected targets:
    if ~np.any( np.logical_and(detected_targets[:, 0] == X_targets[i, 0], detected_targets[:, 1] == X_targets[i, 1]) ):
      undetected_targets = np.vstack([undetected_targets, X_targets[i, :]])
  undetected_targets = np.delete(undetected_targets, 0, 0)
  
  # Find which clusters are false positives:
  false_positives_indexes = (~np.isin( np.arange( len(cluster_centers) ), correct_clusters_indices )).astype(int)
  false_positives_indexes = np.where(false_positives_indexes == 1)
  false_positives = np.array(cluster_centers)[false_positives_indexes]
  
  # Calculate perfomance metrics:
  positives_count = detected_targets.size/2 + undetected_targets.size/2
  precision = (detected_targets.size/2)/(detected_targets.size/2 + false_positives.size/2) if (detected_targets.size > 0 or false_positives.size > 0) else -1
  recall = (detected_targets.size/2)/positives_count
  fpr = (false_positives.size/2)/(6) # number of false positives divided by area (km^2)
  f1_score = 2*precision*recall/(precision + recall) if (precision > 0 or recall > 0) else 0
  return f1_score, precision, recall, fpr, detected_targets, undetected_targets, false_positives









# Given a list of images, execute the whole prediction and classification process and save all obtained information and position of targets.
def predict_and_save(model_conv, model_classconv, img_names, detect_thresh, classif_thresh, save):
  if (type(img_names) != list): img_names = [img_names]
  Pred_mask, Predicted_clusters, cluster_centers = np.empty((len(img_names)), dtype=object), np.empty((len(img_names)), dtype=object), np.empty((len(img_names)), dtype=object)
  tp, fp, fn = np.zeros(len(img_names)), np.zeros(len(img_names)), np.zeros(len(img_names))
  
  # Execute 'detection_performance' function in parallel for all images in 'img_names'. Use 'detect_thresh' and 'classif_thresh' as the predetection binary threshold and classification threshold:
  parallel_detectperf = Parallel(n_jobs = 4, backend = 'threading')(
      delayed(detection_perfomance)(model_conv, model_classconv, 0, detect_thresh, classif_thresh, img_names[i], save) for i in range( len(img_names) ) )
  f1_score, precision, recall, fpr, detected_targets, undetected_targets, false_positives = zip(*parallel_detectperf)
  
  # Generate a table containing the number of detected targets, false positives and undetected targets for each prediction:
  for i, img_name in enumerate(img_names):
    tp[i], fp[i], fn[i] = detected_targets[i].size/2, false_positives[i].size/2, undetected_targets[i].size/2

  print("| TP    FP    FN   Image |")
  print("|-------------------------|")
  for i, img_name in enumerate(img_names):
    print(" %d    %d    %d    %s" % (tp[i], fp[i], fn[i], img_name))
    
  return detected_targets, false_positives







# This function executes the 'detection_perfomance' function for each classification model threshold, specified by 'classif_threshs' array, and plots the ROC curve (if requested).
def roc_classif(model_conv, model_classconv, img_name, img_dataset, detect_thresh, classif_threshs, plot, predict, Pred_mask, kfold):
    detected_targets, undetected_targets, false_positives = np.empty((len(classif_threshs)), dtype=object), np.empty((len(classif_threshs)), dtype=object), np.empty((len(classif_threshs)), dtype=object)
    f1_score, precision, recall, fpr = np.zeros( len(classif_threshs) ), np.zeros( len(classif_threshs) ), np.zeros( len(classif_threshs) ), np.zeros( len(classif_threshs) )
    classif_threshs = np.sort(classif_threshs, axis=None)

    # Call the predetection predictor if requested to get the prediction heatmap:
    if predict == 1:
      Pred_mask = np.reshape( predict_image(model_conv, img_name, 1, img_dataset), (3000, 2000) ) if kfold else np.reshape( predict_image(img_name, 0, 0), (3000, 2000) )
    
    # With the provided prediction heatmap, find cluster, make the classification prediction and calculate the perfomance parameters for all classification thresholds 'classif_threshs':
    for i in range(len( classif_threshs ) ) :
      f1_score[i], precision[i], recall[i], fpr[i], detected_targets[i], undetected_targets[i], false_positives[i] = detection_perfomance(model_conv, model_classconv, Pred_mask, detect_thresh, classif_threshs[i], img_name, 0)
    
    if plot:
      plt.grid(True)
      plt.xlim(np.amin(fpr)*0.85, np.amax(fpr)*1.15)
      plt.ylim(np.amin(recall)*0.85, np.amax(recall)*1.15)
      plt.xlabel("False Alarm Rate (1/km^2)")
      plt.ylabel("Probability of Detection (TPR)")
      plt.plot(fpr, recall, 'go-')
      plt.show()
    best_f1 = np.amax(f1_score)
    best_classif_threshs_index = np.argmax(f1_score)
    best_classif_threshs = classif_threshs[best_classif_threshs_index] 
    print("Best classif_threshs = %.4f, with F1-score = %.4f ." % (best_classif_threshs, best_f1) ) # The f1-score is just informative.
    return Pred_mask, f1_score, precision, recall, fpr, detected_targets, undetected_targets, false_positives, best_classif_threshs, best_classif_threshs_index








# This function executes 'roc_classif' for multiple images and calculates the ROC curve based on the information from all tested images.
def roc_multiple_images(model_conv, model_classconv, img_names, img_dataset, classif_threshs, detect_thresh, kfold):
  best_f1_scores, best_precisions, best_recalls, best_fprs, best_thresholds = np.zeros( len(img_names) ), np.zeros( len(img_names) ), np.zeros( len(img_names) ), np.zeros( len(img_names) ), np.zeros( len(img_names) )
  mean_f1_scores, mean_precisions, mean_recalls, mean_fprs = np.zeros( len(classif_threshs) ), np.zeros( len(classif_threshs) ), np.zeros( len(classif_threshs) ), np.zeros( len(classif_threshs) )

  # Parallelize calls to 'roc_classif' to speed up the process:
  parallel_vars = Parallel(n_jobs=4, backend = 'threading', verbose = 10, max_nbytes = '100M')( delayed(roc_classif)(model_conv, model_classconv, img_names[i], img_dataset, detect_thresh, classif_threshs, 0, 1, 0, kfold) for i in range( len(img_names) ))
  _, f1_score, precision, recall, fpr, _, _, _, best_threshold, best_threshold_index = zip(*parallel_vars)
  mean_f1_scores, mean_precisions, mean_recalls, mean_fprs = np.mean(np.array(f1_score), axis = 0), np.mean(np.array(precision), axis = 0), np.mean(np.array(recall), axis = 0) ,np.mean(np.array(fpr), axis = 0)
  
  # Get worst fpr whose recall equals to max recall:
  lbti = np.argmax(mean_recalls) 

  # Discart thresholds that yielded a lower probability of detection with false alarm rate higher than the threshold(s) with best probability of detection. 
  # This is done in order to discart useless thresholds from the ROC:
  mean_fprs = mean_fprs[lbti:-1]
  mean_recalls = mean_recalls[lbti:-1]
  classif_threshs = classif_threshs[lbti:-1]

  # Print the perfomance parameters for each tested threshold:
  print( np.stack( (classif_threshs, mean_fprs, mean_recalls), axis = 1 ) )
  
  # Plot the ROC:
  plt.grid(True)
  startx, endx, starty, endy = 0, 0.5, 0.91, 1.005
  plt.xlim(startx, endx)
  plt.ylim(starty, endy)
  tickstepx = 0.05
  tickstepy = 0.01
  plt.xticks( np.arange(startx, endx + tickstepx, tickstepx ), fontsize = 8 )
  plt.yticks( np.arange(starty, endy + tickstepy, tickstepy ) )
  plt.xlabel("False Alarm Rate (1/km^2)")
  plt.ylabel("Probabilty of Detection (TPR)")
  plt.plot(mean_fprs, mean_recalls, 'go-', markersize = 3, linewidth = 1)
  for i, txt in enumerate(classif_threshs):
    plt.annotate("%.4f" % txt, (mean_fprs[i], mean_recalls[i]), fontsize = 2)
  # Mark the point from Renato's (Dal Molin) paper:
  plt.plot(0.28, 0.9633, 'bx', markersize = 7)
  plt.annotate("(0.28, 0.9633)", (0.28, 0.9633), fontsize = 7)
  plt.savefig( os.path.join(datab_imgs_path, "predictions/roc.pdf") )  
  plt.show()
  
  return best_f1_scores, best_precisions, best_recalls, best_thresholds, mean_f1_scores, mean_precisions, mean_recalls, mean_fprs






