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
from sklearn import metrics
from scipy.cluster.hierarchy import fclusterdata
from numba import jit, njit
from operator import itemgetter 
from joblib import Parallel, delayed
import multiprocessing
import dill
import tikzplotlib
from scipy import ndimage, misc
from focalloss import *
from basefunctions import *

# Definition of a simple class containing properties about speckle and gaussian noise.
class noiseStruct(object):
  def __init__(self, speckle_var, gaussian_var, aug_negat_prob, rot_angle):
    self.gaussian_mean = 0
    self.speckle_mean = 1
    self.speckle_var = speckle_var
    self.gaussian_var = gaussian_var
    self.aug_negat_prob = aug_negat_prob
    self.rot_angle = rot_angle

# This function adds speckle (multiplicative) and gaussian noise to a matrix image. Used to augment the database.
def add_speckle_noise_to_img(Image, noiseprop):
  Speckle_noise = np.random.normal(loc = noiseprop.speckle_mean, scale = np.sqrt(noiseprop.speckle_var), size = Image.shape)
  Gaussian_noise = np.random.normal(loc = noiseprop.gaussian_mean, scale = np.sqrt(noiseprop.gaussian_var), size = Image.shape)
  Image = ndimage.rotate(Image, noiseprop.rot_angle, reshape=False, mode = 'wrap')
  Image = np.floor(Image*Speckle_noise + Gaussian_noise)
  return Image


def gen_classification_gt_v2(Images_vector, img_name, window_size, overlap, noiseproperties_vec):
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
  Y_targets = Y_targets[ Y_targets[:, 0] == 1 , :]
  nextwindow = 0
  
  print("Windows to process: " + str(nblocks_vert*nblocks_horiz) )

    # Cutting windows with size = 'window_size' and overlap = 'overlap'. A padding of zeros is added to the border windows as needed.
    # Collect negative examples and ignore positive windows.
  for wind_v, j in enumerate( range(0, npixels_vert, effective_wsize) ):
    if (j % 30*npixels_vert == 0): print("Processing window " + str(wind_v*nblocks_horiz) )
    for wind_h, k in enumerate( range(0, npixels_horiz, effective_wsize) ):
      if ( (k - overlap >= 0) and (j - overlap >= 0) and (k + effective_wsize <= im_dims[1]) and (j + effective_wsize <= im_dims[0]) ):
        vert_range = np.arange((j-overlap), (j+effective_wsize))
        horiz_range = np.arange((k-overlap), (k+effective_wsize))
        window_points = np.transpose([np.tile(vert_range, len(horiz_range)), np.repeat(horiz_range, len(vert_range))])
        target = np.any( [np.any( ( X_targets == window_points[i, :]).all( axis = 1 ) ) for i in range(window_points.shape[0] ) ] )
        if not target:
          X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, :, :] = np.reshape( Images_vector[window_points[:, 0], window_points[:, 1]], (window_size, window_size), order='F'  )
          # Y_class_window[wind_h + wind_v*nblocks_horiz] = 0
        else:
          X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, :, :] = X_full_pixval_class_window[wind_h - 1 + wind_v*nblocks_horiz, :, :]
          # Y_class_window[wind_h + wind_v*nblocks_horiz] = 1
      else:
        for wpos_v, l in enumerate( range(-overlap, effective_wsize) ):
          for wpos_h, m in enumerate( range(-overlap, effective_wsize) ):
            if ( (k+m >= 0) and (j+l >= 0) and (k+m < im_dims[1]) and (j+l < im_dims[0]) ):
              X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, wpos_v, wpos_h] = Images_vector[ j + l, k + m]
              if np.any( ( X_targets == np.array([j + l, k + m]) ).all( axis = 1 ) ): 
                X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, :, :] = X_full_pixval_class_window[wind_h - 1 + wind_v*nblocks_horiz, :, :]
                # Y_class_window[wind_h + wind_v*nblocks_horiz] = 1
                nextwindow = 1
                break
            else:
              X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, wpos_v, wpos_h] = 0
          if nextwindow: 
            nextwindow = 0
            break        

  halfwindow = int( window_size/2 )
  for i in range( X_targets.shape[0] ):
    unparity = 0 if halfwindow == window_size/2 else 1
    cluster_center = np.array([ X_targets[i, 0], X_targets[i, 1] ])
    # Determine the limit coordinates of the window to be analized.
    lowbound_horiz = -halfwindow + cluster_center[1]
    highbound_horiz = halfwindow + cluster_center[1]
    
    lowbound_vert = -halfwindow + cluster_center[0]
    highbound_vert = halfwindow + cluster_center[0]

    Pred_window = np.zeros( ( 1, window_size, window_size ) )
    Pred_class = np.ones( (1, 1) )
    # If the coordinates are not out of bound, a vectorized window extraction takes place. 
    if (lowbound_horiz >= 0) and (lowbound_vert >= 0) and (highbound_horiz < im_dims[1]) and (highbound_vert < im_dims[0]):
      Pred_window[0, :, :] = Images_vector[lowbound_vert:highbound_vert, lowbound_horiz:highbound_horiz]
    else:
      # The windows that, at some point, exceed the image's bounds are padded with zeros:
      for i, i_vert in enumerate( range(-halfwindow, halfwindow + unparity) ):
        for j, j_horiz in enumerate( range(-halfwindow, halfwindow + unparity) ):
          if ( (i_vert + cluster_center[0]) >= 0 ) and ( (i_vert + cluster_center[0]) < im_dims[0] ) and ( (j_horiz + cluster_center[1]) >= 0) and ( (j_horiz + cluster_center[1]) < im_dims[1]) :
            Pred_window[0, i, j] = Images_vector[ i_vert + cluster_center[0], j_horiz + cluster_center[1] ]
          else:
            Pred_window[0, i, j] = 0

    X_full_pixval_class_window = np.concatenate( [X_full_pixval_class_window, Pred_window], axis = 0 )
    Y_class_window = np.concatenate([Y_class_window, Pred_class] , axis = 0)
  
  addnoise = 0
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
            X_full_pixval_class_window[i_block, :, :] = add_speckle_noise_to_img(X_full_pixval_class_window[i_block, :, :], noiseproperties_vec[i_noisevec])
        elif int(block_class) == 1:
          X_noisypositives[j_pos, :, :] = add_speckle_noise_to_img(X_full_pixval_class_window[i_block, :, :], noiseproperties_vec[i_noisevec])
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
  parallel_genclassgt = Parallel(n_jobs = num_cores, backend='multiprocessing')(delayed(gen_classification_gt_v2)(Images_vector[i], img_names[i], window_size, overlap, noiseprop_vec) for i in range( n_images ) )
  X_full_pixval_class_window, Y_class_window = zip(*parallel_genclassgt)

  # Save all windows as single variable
  if save:
    with open(datab_imgs_path + 'classification_data/' + 'xy_classification.pkl' , 'wb') as f:  # Python 3: open(..., 'wb')
      pickle.dump([X_full_pixval_class_window, Y_class_window], f)
      print('saved')

  return X_full_pixval_class_window, Y_class_window




# This function loads from disk the variables containing all windows created for classification training/validation. 
def load_multiple_classification_gt():
  # Load all windows that went through the noise data augmentation:
  with open(datab_imgs_path + 'classification_data/' + 'xy_classification.pkl' , 'rb') as f:  # Python 3: open(..., 'wb')
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
  save = 1
  Image_vec_norm = standardize_pixels( bwimg( os.path.join(datab_imgs_path, img_name + ".jpg") ) )
  Image_vec_norm = np.reshape(Image_vec_norm, (1, Image_vec_norm.shape[0], Image_vec_norm.shape[1], 1) )
  if kfold:
    n_split = len(whole_set)//4
    n_images = len(whole_set)
    kfold_img_index = int( (whole_set.index(img_name) // (n_images // n_split) )) 
    print(img_name, kfold_img_index)
    # Pass the image through the predetection model:
    Target_map = np.reshape( model_conv[kfold_img_index].predict(Image_vec_norm, verbose = 1), (3000, 2000) )
  else:
    Target_map = np.reshape( model_conv.predict(Image_vec_norm, verbose = 1), (3000, 2000) )
  if save:
    Map_255 = (Target_map + np.amin(Target_map) )/np.amax(Target_map)*255
    saveimg(Map_255, os.path.join(datab_imgs_path, "predictions/" + img_name + "_predetectmap.jpg"))
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
  ctlen = len(classif_thresh)
  predetect_only = 0
  n_split = len(trainval_dataset)//4
  n_images = len(trainval_dataset)
  try:
    image_index = trainval_dataset.index(img_name)
    kfold_index = int( image_index // (n_images // n_split) )
  except ValueError:
    image_index = whole_set.index(img_name)
    kfold_index = 0
  Pred_mask = Pred > detect_thresh # Generate binary map.
  if save:
    Mask = np.reshape(255*Pred_mask, (3000, 2000)).astype('uint8')
    saveimg(Mask, os.path.join(datab_imgs_path, "predictions/" + img_name + "_predetectbinary.jpg"))
  candidates_coordinates = np.transpose( np.nonzero(Pred_mask) )
  candidates_predval = Pred[candidates_coordinates[:, 0], candidates_coordinates[:, 1]] # Will be used as weights.
  candidates_predval /= np.mean(candidates_predval) # Normalize by mean.
  clusters = DBSCAN(eps = 5.01, min_samples = 8, n_jobs = -1 ).fit(candidates_coordinates) # Run DBSCAN. Maximum distance between pixels is of 5px. 8 samples are needed to form a cluster.
  unique_labels = np.array(list( set(clusters.labels_) ) ) # Convert to ndarray.
  unique_nonoise = unique_labels[unique_labels != -1] # Retrieve the number of found clusters
  
  possible_clusters = np.zeros( ( len(unique_nonoise), 2 ) , dtype = 'int32')
  possible_clusters_probs = np.zeros( ( len(unique_nonoise) ) )
  clusters_centers = np.empty(ctlen, dtype=object)
  for i in range(ctlen): clusters_centers[i] = []
  
  # Calculate found clusters' centers and feed to the classification model. For each cluster, if the result is bigger than a threshold, the cluster is assumed to be a target.
  j = 0
  # print("Number of clusters to analyze:" + str(unique_nonoise.shape) )
  for k in set(unique_nonoise):
    class_member_mask = (clusters.labels_ == k)
    class_members = candidates_coordinates[class_member_mask, :]
    possible_clusters[j, :] = np.average( class_members, axis = 0, weights = np.power(candidates_predval[class_member_mask], 2 ) ).astype('int32') # For each cluster, calculates the average position of all cluster members, weighted by the heatmap value of each member. 
    possible_clusters_probs[j] = np.reshape( predict_class(model_classconv, img_name, trainval_dataset, possible_clusters[j, :]), 1)
    for i in range(ctlen):
      if possible_clusters_probs[j] > classif_thresh[i] or predetect_only:
        clusters_centers[i].append( possible_clusters[j, :]  ) # If the classificator returns a value bigger than 'classif_thresh', the cluster will be assumed to be a target.
    j += 1

  

  # # Final_mask = np.zeros( Pred.shape , dtype = np.uint8)
  # if save:
  #   Final_mask = bwimg(os.path.join(datab_imgs_path, img_name + ".jpg")).astype(np.uint8)
  #   Final_mask = np.repeat( np.reshape(Final_mask, (Final_mask.shape[0], Final_mask.shape[1], 1) ), repeats = 3, axis = 2 )
  #   for i, cluster in enumerate( clusters_centers ):
  #     Final_mask = cv2.circle(Final_mask, (cluster[1], cluster[0]), radius = 10, color = (50, 255, 0), thickness = 2)
  
  #   # showimg(Final_mask)
  #   plt.imshow(Final_mask)
    # saveimg(Final_mask, os.path.join(datab_imgs_path, "predictions/" + img_name + "_dbscan_twosteps_prediction.jpg") )
    # np.savetxt(os.path.join(datab_imgs_path, "predictions/" + img_name + "_cluster_centers_prediction.txt"), clusters_centers, "%d" )
  
  return clusters_centers







# This function recieves the predetection heatmap, calls the cluster localization function 'find_clusters_andsave' and calculates the detection performance indicators (true positive rate, false alarm rate, etc).
def detection_perfomance(model_conv, model_classconv, Pred, detect_thresh, classif_thresh, img_name, save):
  ctlen = len(classif_thresh)
  correct_clusters_indices, detected_targets, undetected_targets, false_positives, correct_clusters_indices = np.empty(ctlen, dtype=object), np.empty(ctlen, dtype=object), np.empty(ctlen, dtype=object), np.empty(ctlen, dtype=object), np.empty(ctlen, dtype=object)
  for i in range(ctlen): 
    correct_clusters_indices[i] = np.zeros(1)
    detected_targets[i] = np.zeros((1, 2))
    undetected_targets[i] = np.zeros((1, 2))
    false_positives[i] = np.zeros((1, 2))
    correct_clusters_indices[i] = np.zeros((1, 1))
  
  Img_GT_mask = np.zeros((3000, 2000))

  if (isinstance(Pred, int) or isinstance(Pred, float)):
    Pred = predict_image(model_conv, img_name, 1, whole_set) # Generates the predetection heatmap if none is provided.

  Pred_mask = Pred > detect_thresh # Apply detection threshold to generate the binary map.
  # print(np.sum(Pred_mask))
  # Retreive clusters from predetection heatmap:
  cluster_centers = find_clusters_andsave(model_classconv, Pred, detect_thresh, classif_thresh, img_name, whole_set, save)
  # Get ground truth position of target pixels:
  tmp_mat = np.loadtxt(datab_imgs_path + "labels/" + img_name + ".txt", delimiter = ' ', usecols = range(4))
  tmp_mat = tmp_mat.astype(int)
  X_targets = tmp_mat[:, 0:2]
  Y_targets = tmp_mat[:, 2:4]
  X_targets = X_targets[ Y_targets[:, 0] == 1 , :]
  Y_targets = Y_targets[ Y_targets[:, 0] == 1 , :]

  # For each found cluster (possible target), calculate the distance from all true targets based on the ground truth:
  for h in range(ctlen):
    for i in range(X_targets.shape[0]): 
      for j in range(len(cluster_centers[h]) ):
        # If a cluster is closer than 10px of a ground truth position, the cluster is declared to be a correct detection.
        distance_from_target = np.sqrt( (cluster_centers[h][j][0] - X_targets[i][0])**2 + (cluster_centers[h][j][1] - X_targets[i][1])**2 )
        if distance_from_target < 10: #10px = 10m
          correct_clusters_indices[h] = np.append(correct_clusters_indices[h], j)
          detected_targets[h] = np.vstack([detected_targets[h], X_targets[i, :] ] )
          break

  for h in range(ctlen): 
    detected_targets[h] = np.delete(detected_targets[h], 0, 0) # Delete first useless line.
    correct_clusters_indices[h]= np.delete(correct_clusters_indices[h], 0, 0) # Delete first useless line.
    for i in range(X_targets.shape[0]): # Find undetected targets:
      if ~np.any( np.logical_and(detected_targets[h][:, 0] == X_targets[i, 0], detected_targets[h][:, 1] == X_targets[i, 1]) ):
        undetected_targets[h] = np.vstack([undetected_targets[h], X_targets[i, :]])
    undetected_targets[h] = np.delete(undetected_targets[h], 0, 0)

  
  
  # Find which clusters are false positives:
  for h in range(ctlen):
    false_positives_indexes = (~np.isin( np.arange( len(cluster_centers[h]) ), correct_clusters_indices[h] )).astype(int)
    false_positives_indexes = np.where(false_positives_indexes == 1)
    false_positives[h] = np.array(cluster_centers[h])[false_positives_indexes]  
  # Plot and save
  if save:
    Final_mask = bwimg(os.path.join(datab_imgs_path, img_name + ".jpg")).astype(np.uint8)
    Final_mask = np.repeat( np.reshape(Final_mask, (Final_mask.shape[0], Final_mask.shape[1], 1) ), repeats = 3, axis = 2 )
    # Draw true detections
    for i, cluster in enumerate( detected_targets[0]):
      Final_mask = cv2.rectangle(Final_mask, pt1 = (int(cluster[1])-24, int(cluster[0])-24), pt2 = (int(cluster[1])+23, int(cluster[0])+23), color = (255, 0, 255), thickness = 1)
      # Final_mask = cv2.rectangle(Final_mask, pt1 = (int(cluster[1])-12, int(cluster[0])-12), pt2 = (int(cluster[1])+12, int(cluster[0])+12), color = (0, 255, 0), thickness = 2)
    # Draw false alarms
    for i, cluster in enumerate( false_positives[0] ):
      continue
      # Final_mask = cv2.rectangle(Final_mask, pt1 = (int(cluster[1])-24, int(cluster[0])-24), pt2 = (int(cluster[1])+23, int(cluster[0])+23), color = (255, 0, 255), thickness = 1)
      # Final_mask = cv2.rectangle(Final_mask, pt1 = (int(cluster[1])-12, int(cluster[0])-12), pt2 = (int(cluster[1])+12, int(cluster[0])+12), color = (255, 0, 0), thickness = 2)
     
    # Draw undetected targets
    for i, cluster in enumerate( undetected_targets[0] ):
      Final_mask = cv2.rectangle(Final_mask, pt1 = (int(cluster[1])-24, int(cluster[0]-24)), pt2 = (int(cluster[1])+23, int(cluster[0]+23)), color = (255, 255, 20), thickness = 2)
      # continue
    plt.imshow(Final_mask)
    saveimg(Final_mask, os.path.join(datab_imgs_path, "predictions/" + img_name + "_twosteps_prediction.jpg") )
    np.savetxt(os.path.join(datab_imgs_path, "predictions/" + img_name + "_cluster_centers_prediction.txt"), cluster_centers[0], "%d" )
  
  # Calculate perfomance metrics:
  positives_count, precision, recall, fpr, f1_score = np.zeros(ctlen), np.zeros(ctlen), np.zeros(ctlen), np.zeros(ctlen), np.zeros(ctlen)
  for h in range(ctlen):
    positives_count[h] = detected_targets[h].size/2 + undetected_targets[h].size/2
    precision[h] = (detected_targets[h].size/2)/(detected_targets[h].size/2 + false_positives[h].size/2) if (detected_targets[h].size > 0 or false_positives[h].size > 0) else -1
    recall[h] = (detected_targets[h].size/2)/positives_count[h]
    fpr[h] = (false_positives[h].size/2)/(6) # number of false positives divided by area (km^2)
    f1_score[h] = 2*precision[h]*recall[h]/(precision[h] + recall[h]) if (precision[h] > 0 or recall[h] > 0) else 0
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
  
  #Generate a table containing the number of detected targets, false positives and undetected targets for each prediction:
  for i, img_name in enumerate(img_names):
    tp[i], fp[i], fn[i] = detected_targets[i][0].size/2, false_positives[i][0].size/2, undetected_targets[i][0].size/2
  print("Classification Threshold = %.2f" % classif_thresh[0])
  print("| TP    FP    FN   Image |")
  print("|-------------------------|")
  for i, img_name in enumerate(img_names):
    print(" %d    %d    %d    %s" % (tp[i], fp[i], fn[i], img_name))
  print("Total Pd=%.4f \n Total FAR=%.4f" % (np.sum(tp)/( (tp[0] + fn[0])*len(img_names) ), np.sum(fp)/(6*len(img_names) ) ) )
  return detected_targets, false_positives







# This function executes the 'detection_perfomance' function for each classification model threshold, specified by 'classif_threshs' array, and plots the ROC curve (if requested).
def roc_classif(model_conv, model_classconv, img_name, img_dataset, detect_thresh, classif_threshs, predict, Pred_mask, kfold):
    # detected_targets, undetected_targets, false_positives = np.empty((len(classif_threshs)), dtype=object), np.empty((len(classif_threshs)), dtype=object), np.empty((len(classif_threshs)), dtype=object)
    # f1_score, precision, recall, fpr = np.zeros( len(classif_threshs) ), np.zeros( len(classif_threshs) ), np.zeros( len(classif_threshs) ), np.zeros( len(classif_threshs) )
    classif_threshs = np.sort(classif_threshs, axis=None)

    # Call the predetection predictor if requested to get the prediction heatmap:
    if predict == 1:
      Pred_mask = np.reshape( predict_image(model_conv, img_name, 1, img_dataset), (3000, 2000) ) if kfold else np.reshape( predict_image(img_name, 0, 0), (3000, 2000) )
    
    # With the provided prediction heatmap, find cluster, make the classification prediction and calculate the perfomance parameters for all classification thresholds 'classif_threshs':
    # for i in range(len( classif_threshs ) ) :
      # f1_score[i], precision[i], recall[i], fpr[i], detected_targets[i], undetected_targets[i], false_positives[i] = detection_perfomance(model_conv, model_classconv, Pred_mask, detect_thresh, classif_threshs[i], img_name, 0)
    
    f1_score, precision, recall, fpr, detected_targets, undetected_targets, false_positives = detection_perfomance(model_conv, model_classconv, Pred_mask, detect_thresh, classif_threshs, img_name, 0)
    best_f1 = np.amax(f1_score)
    best_classif_threshs_index = np.argmax(f1_score)
    best_classif_threshs = classif_threshs[best_classif_threshs_index] 
    print("Best classif_threshs = %.4f, with F1-score = %.4f ." % (best_classif_threshs, best_f1) ) # The f1-score is just informative.
    return Pred_mask, f1_score, precision, recall, fpr, detected_targets, undetected_targets, false_positives




# This function executes 'roc_classif' for multiple images and calculates the ROC curve based on the information from all tested images.
def roc_multiple_images(model_conv, model_classconv, img_names, img_dataset, classif_threshs, detect_threshs, kfold):

  classif_threshs, detect_threshs = np.array(classif_threshs), np.array(detect_threshs)
  mean_f1_scores, mean_precisions, mean_recalls, mean_fprs = np.empty( len(detect_threshs), dtype=object), np.empty( len(detect_threshs), dtype=object), np.empty( len(detect_threshs), dtype=object), np.empty( len(detect_threshs), dtype=object)
  num_cores = multiprocessing.cpu_count() if ( len(classif_threshs) >  multiprocessing.cpu_count()) else len(classif_threshs)
  performance_matrix = np.zeros( ( len(detect_threshs)*len(classif_threshs), 4 ) )
  auc = np.zeros(len(detect_threshs))
  for j in range( len(detect_threshs) ):
    # Parallelize calls to 'roc_classif' to speed up the process:
    parallel_vars = Parallel(n_jobs = 1, backend = 'sequential', verbose = 10)( delayed(roc_classif)(model_conv, model_classconv, img_names[i], img_dataset, detect_threshs[j], classif_threshs, 1, 0, kfold) for i in range( len(img_names) ) )
    _, f1_score, precision, recall, fpr, _, _, _ = zip(*parallel_vars)
    mean_precisions[j], mean_recalls[j], mean_fprs[j] = np.mean(np.array(precision), axis = 0), np.mean(np.array(recall), axis = 0), np.mean(np.array(fpr), axis = 0)
    mean_f1_scores[j] =  2*(mean_precisions[j]*mean_recalls[j])/( mean_precisions[j] + mean_recalls[j] ) 
    # Matrix with columns (detect threshold, classif threshold, FPR, recall)
    performance_matrix[(j*len(classif_threshs)):(j*len(classif_threshs) + len(classif_threshs)), :] = np.stack( (detect_threshs[j]*np.ones( len(classif_threshs) ), classif_threshs, mean_fprs[j], mean_recalls[j] ), axis = 1)
    # Print the perfomance parameters for each tested threshold:
    print("")
    print("Predetection Threshold = " + str(detect_threshs[j]) )
    print( np.stack( (classif_threshs, mean_fprs[j], mean_recalls[j]), axis = 1 ) )
    print("Max F1-Score = " + str( np.amax(mean_f1_scores[j]) ) + " for classif threshold = " + str( classif_threshs[ np.argmax(mean_f1_scores[j]) ] ) )
    zerop8_index = (mean_fprs[j] <= 0.8).nonzero()
    zerop8_index = zerop8_index[0][0]
    print(zerop8_index)
    fpr_aux, recall_aux = mean_fprs[j][zerop8_index:], mean_recalls[j][zerop8_index:]
    recall_aux = np.insert(recall_aux, 0, recall_aux[0]) if fpr_aux[0] < 0.8 else recall_aux
    fpr_aux = np.insert(fpr_aux, 0, 0.8) if fpr_aux[0] < 0.8 else fpr_aux
    print(fpr_aux)
    print(recall_aux)
    auc[j] = metrics.auc(fpr_aux, recall_aux)
    print('The AUC (FAR=[0, 0.8]) for the segmentation threshold %.4f is equal to %.5f .\n' % (detect_threshs[j], auc[j]) )

  # performance_matrix = np.sort( performance_matrix, axis = -1 )
  print("Perfomance Matrix:")
  print("(detect threshold, classif threshold, FPR, recall)")
  print(performance_matrix)
  


  roc_matrix = np.zeros((0, 4))
  
  for i_perf in range( performance_matrix.shape[0] ):
    badpointflag = 0
    for j_compar in range( performance_matrix.shape[0] ):
      if ( performance_matrix[j_compar, 2] < performance_matrix[i_perf, 2] ) and ( performance_matrix[j_compar, 3] >= performance_matrix[i_perf, 3] ):
        badpointflag = 1
        break
    if not badpointflag:
      roc_matrix = np.vstack( ( roc_matrix, performance_matrix[i_perf, :] ) )
  
    
  # roc_matrix = np.sort( roc_matrix, axis = -1 )

  print("ROC Matrix:")
  print("(detect threshold, classif threshold, FPR, recall)")
  print(roc_matrix)
  
  
  # Plot the ROC:
  point_colors = np.linspace(0, 1, len(detect_threshs) )

  
  rcParams['figure.figsize'] = [12, 12]
  plt.grid(True)
  startx, endx, starty, endy = 0, 1.52, 0.86, 1.005
  plt.xlim(startx, endx)
  plt.ylim(starty, endy)
  tickstepx = 0.1
  tickstepy = 0.01
  plt.xticks( np.arange(startx, endx + tickstepx/2, tickstepx ), fontsize = 10 )
  plt.yticks( np.arange(starty, endy + tickstepy/2, tickstepy ), fontsize = 12 )
  plt.xlabel("False Alarm Rate [$1/\mathrm{km}^2$]", fontsize = 15)
  plt.ylabel("Probabilty of Detection", fontsize = 15)
  annotate_y = 4
  
  # for j in range( len(roc_matrix) ):
  ##   print( point_colors[ detect_threshs == roc_matrix[j, 0] ] )
  #   plt.plot(roc_matrix[j, 2], roc_matrix[j, 3], 'o-', markersize = 8, markeredgecolor = 'k', linewidth = 2, color = plt.cm.RdYlBu( float( point_colors[ detect_threshs == roc_matrix[j, 0] ] ) ) )
  #   plt.annotate("(%.3f, %.3f)" % (roc_matrix[j, 0], roc_matrix[j, 1] ), (roc_matrix[j, 2], roc_matrix[j, 3]), fontsize = 7, xytext = (2, annotate_y), textcoords="offset points")
  #   annotate_y = - annotate_y
  
  plt.plot(roc_matrix[:, 2], roc_matrix[:, 3], linewidth = 2, marker = 'o', markersize = 3, color = 'royalblue', label = 'Proposed Architecture')

  # Mark the point from Renato's (Dal Molin) paper:
  plt.plot(0.28, 0.9633, 'bx', markersize = 5)
  plt.annotate("Dal Molin Jr's 2019 \n Performance", (0.28, 0.9633), xytext = (2, -annotate_y), fontsize = 8, textcoords="offset points")
  
  plt.plot(0.67, 0.97, 'kx', markersize = 5)
  plt.annotate("Lundberg's 2006 \n Performance", (0.67, 0.97), xytext = (2, -annotate_y), fontsize = 8, textcoords="offset points")
  
  vuwave_perf = np.array([ [0.0508862, 0.8704] ,
                            [0.07513, 0.9007], 
                            [0.10078595, 0.9242],
                            [0.11948132, 0.9371], 
                            [0.15779746, 0.95],
                            [0.20459735, 0.9626],
                            [0.33288943, 0.9714], 
                            [0.60145065, 0.983], 
                            [1.4118872, 0.986] ])
  plt.plot(vuwave_perf[:, 0], vuwave_perf[:, 1], color = 'brown', marker = '*',
  markersize = 3, linewidth = 2, linestyle = ':', label = "Vu's 2017 \n Performance" )

  gpalm_perf = np.array([[0.11337048, 0.9413    ],
                        [0.14122123, 0.9597    ],
                        [0.16741717, 0.9696    ],
                        [0.19842665, 0.9774    ],
                        [0.26613378, 0.9794    ],
                        [0.42442406, 0.9813    ],
                        [0.60242086, 0.9822    ],
                        [0.84898497, 0.9841    ],
                        [1.06831666, 0.9844    ],
                        [1.51077573, 0.9863    ]])
  plt.plot(gpalm_perf[:, 0], gpalm_perf[:, 1], color = 'seagreen', marker = '+',
  markersize = 3, linewidth = 2, linestyle = '--', label = "G. Palm's 2020 \n Performance")

  plt.legend(loc = 'lower right')
  plt.savefig( os.path.join(datab_imgs_path, "predictions/roc.pdf") ) 
  # tikzplotlib.clean_figure()
  tikzplotlib.save(os.path.join(datab_imgs_path, "predictions/roc.tex"))
  plt.show()
  
  return mean_f1_scores, mean_precisions, mean_recalls, mean_fprs








# This function cuts the specified image vector into multiple square windows to be used in training and validation. 
# # Data augmentation with noise also takes place below.
# def gen_classification_gt(Images_vector, img_name, window_size, overlap, noiseproperties_vec):
  # onlycenter = 0
  # print(img_name)
  # effective_wsize = window_size - overlap
  # npixels_vert = im_dims[0]
  # npixels_horiz = im_dims[1]
  
  # nblocks_vert = int( ( npixels_vert )/effective_wsize + 1) 
  # nblocks_horiz = int( ( npixels_horiz)/effective_wsize + 1)
  # X_full_pixval_class_window = np.zeros( (nblocks_vert*nblocks_horiz, window_size, window_size) )
  # Y_class_window = np.zeros( (nblocks_vert*nblocks_horiz, 1) )


  # # Retrieves the ground truth of the specified image.
  # tmp_mat = np.loadtxt(datab_imgs_path + "labels/" + img_name + ".txt", delimiter = ' ', usecols = range(4))
  # tmp_mat = tmp_mat.astype(int)
  # X_targets = tmp_mat[:, 0:2]
  # Y_targets = tmp_mat[:, 2:4]
  # X_targets = X_targets[ Y_targets[:, 0] == 1 , :]
  # Y_targets = Y_targets[ Y_targets[:, 0] == 1 , :]

  # print("Windows to process: " + str(nblocks_vert*nblocks_horiz) )

  # if onlycenter:
    # # Cutting windows with size = 'window_size' and overlap = 'overlap'. A padding of zeros is added to the border windows as needed.
    # for wind_v, j in enumerate( range(0, npixels_vert, effective_wsize) ):
      # if (j % 30*npixels_vert == 0): print("Processing window " + str(wind_v*nblocks_horiz) )
      # for wind_h, k in enumerate( range(0, npixels_horiz, effective_wsize) ):
        # if ( (k - overlap >= 0) and (j - overlap >= 0) and (k + effective_wsize <= im_dims[1]) and (j + effective_wsize <= im_dims[0]) ):
          # vert_range = np.arange((j-overlap), (j+effective_wsize))
          # horiz_range = np.arange((k-overlap), (k+effective_wsize))
          # window_points = np.transpose([np.tile(vert_range, len(horiz_range)), np.repeat(horiz_range, len(vert_range))])
          # X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, :, :] = np.reshape( Images_vector[window_points[:, 0], window_points[:, 1]], (window_size, window_size), order='F'  )
          # target = np.any( [np.any( ( X_targets == window_points[i, :]).all( axis = 1 ) ) for i in range(window_points.shape[0] ) ] )
          # Y_class_window[wind_h + wind_v*nblocks_horiz] = target
        # else:
          # for wpos_v, l in enumerate( range(-overlap, effective_wsize) ):
            # for wpos_h, m in enumerate( range(-overlap, effective_wsize) ):
              # if ( (k+m >= 0) and (j+l >= 0) and (k+m < im_dims[1]) and (j+l < im_dims[0]) ):
                # X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, wpos_v, wpos_h] = Images_vector[ j + l, k + m]
                # if np.any( ( X_targets == np.array([j + l, k + m]) ).all( axis = 1 ) ): 
                  # Y_class_window[wind_h + wind_v*nblocks_horiz] = 1
              # else:
                # X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, wpos_v, wpos_h] = 0
  # else:
    # numpixfortarget = np.array( [3**2, 5**2] ) 
    # X_targets, Y_targets = np.reshape(X_targets, ( X_targets.shape[0], X_targets.shape[1], 1 ) ), np.reshape(Y_targets, ( Y_targets.shape[0], Y_targets.shape[1], 1 ) )
    # X_targets, Y_targets = insert_pixels_around_annotations(X_targets, Y_targets)
    # X_targets, Y_targets = np.reshape(X_targets, (X_targets.shape[0], X_targets.shape[1] ) ), np.reshape(Y_targets, (Y_targets.shape[0], Y_targets.shape[1] ) )
    # for wind_v, j in enumerate( range(0, npixels_vert, effective_wsize) ):
      # if (wind_v % 3 == 0): print("Processing window " + str(wind_v*nblocks_horiz) )
      # for wind_h, k in enumerate( range(0, npixels_horiz, effective_wsize) ):
        # if ( (k - overlap >= 0) and (j - overlap >= 0) and (k + effective_wsize <= im_dims[1]) and (j + effective_wsize <= im_dims[0]) ):
          # vert_range = np.arange((j-overlap), (j+effective_wsize))
          # horiz_range = np.arange((k-overlap), (k+effective_wsize))
          # window_points = np.transpose([np.tile(vert_range, len(horiz_range)), np.repeat(horiz_range, len(vert_range))])
          # X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, :, :] = np.reshape( Images_vector[window_points[:, 0], window_points[:, 1]], (window_size, window_size), order='F'  )
          # sumtargetpoints = np.sum( [np.any( ( X_targets == window_points[i, :]).all( axis = 1 ) ) for i in range(window_points.shape[0] ) ] )
          # target = 1 if ( sumtargetpoints == numpixfortarget[0] or sumtargetpoints == numpixfortarget[1]) else 0
          # Y_class_window[wind_h + wind_v*nblocks_horiz] = target
        # else:
          # sumtargetpoints = 0
          # for wpos_v, l in enumerate( range(-overlap, effective_wsize) ):
            # for wpos_h, m in enumerate( range(-overlap, effective_wsize) ):
              # if ( (k+m >= 0) and (j+l >= 0) and (k+m < im_dims[1]) and (j+l < im_dims[0]) ):
                # X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, wpos_v, wpos_h] = Images_vector[ j + l, k + m]
                # targetpoint = np.any( ( X_targets == np.array([j + l, k + m]) ).all( axis = 1 ) ) 
                # if targetpoint:
                  # sumtargetpoints += 1
              # else:
                # X_full_pixval_class_window[wind_h + wind_v*nblocks_horiz, wpos_v, wpos_h] = 0
          # if ( sumtargetpoints == numpixfortarget[0] or sumtargetpoints == numpixfortarget[1]):
            # print(sumtargetpoints)
            # Y_class_window[wind_h + wind_v*nblocks_horiz] = 1
          # else:
            # Y_class_window[wind_h + wind_v*nblocks_horiz] = 0

  # addnoise = 0
  # if addnoise:
    # print("Adding multiplicative and additive noise radomly to nontarget windows. Augmenting database with noisy target windows.")

    # # Adding noise to the negative windows with a rate of 'noiseproperties_vec[i_noisevec].aug_negat_prob'. Noisy positive windows are added to the dataset,
    # # mantaining all non-noisy positive windows.
    # npositives = np.sum(Y_class_window[:, 0] == 1)*len(noiseproperties_vec)
    # X_noisypositives = np.zeros( ( npositives, X_full_pixval_class_window.shape[1], X_full_pixval_class_window.shape[2] ) )
    # Y_noisypositives = np.ones( (npositives, 1) )
    # j_pos = 0
    # for i_block, block_class in enumerate(Y_class_window[:, 0]):
      # for i_noisevec in range(len(noiseproperties_vec)):
        # # print("Speckle variance = " + str(noiseproperties_vec[i_noisevec].speckle_var) + ". Gaussian variance = " + str(noiseproperties_vec[i_noisevec].gaussian_var) + "." )
        # if int(block_class) == 0:
          # # Generate Bernoulli RV in order to choose by chance the negative images to which noise will be added:
          # addnegnoise = scipy.stats.bernoulli.rvs(noiseproperties_vec[i_noisevec].aug_negat_prob)
          # if addnegnoise:
            # X_full_pixval_class_window[i_block, :, :] = add_speckle_noise_to_img(X_full_pixval_class_window[i_block, :, :], noiseproperties_vec[i_noisevec])
        # elif int(block_class) == 1:
          # X_noisypositives[j_pos, :, :] = add_speckle_noise_to_img(X_full_pixval_class_window[i_block, :, :], noiseproperties_vec[i_noisevec])
          # j_pos += 1
    # X_full_pixval_class_window = np.concatenate((X_full_pixval_class_window, X_noisypositives), axis = 0)
    # Y_class_window = np.concatenate( (Y_class_window, Y_noisypositives), axis = 0 )
    
  # # Standardizing all resulting windows.
  # for i_window in range( X_full_pixval_class_window.shape[0] ):
    # X_full_pixval_class_window[i_window, :, :] = standardize_pixels(X_full_pixval_class_window[i_window, :, :])
  
  # return X_full_pixval_class_window, Y_class_window


