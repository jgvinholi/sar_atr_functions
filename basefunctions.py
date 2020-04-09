def bwimg(impath): # Get black and white image array
    return np.mean(cv2.imread(impath), axis = 2)

def showimg(img): # Displays image
    plt.imshow(img, cmap="gray")
    return

def saveimg(img, path): # Saves image
    # print(path)
    cv2.imwrite(path, img)  
    return

def varmean_norm(img): # Returns normalized image
    img_mean, img_var = np.mean(img), np.var(img)
    img_norm = (img - img_mean)/np.sqrt(img_var) # Normalize image by mean and variance (zero mean and 1 variance)
    return img_norm

def dif(img1, img2):
    return img1 - img2

def dif2(img1, img2):
    return (img1 - img2)/2 + 128

def dif2_avg(img1, img2): # Calculates the difference image with mean 128
    diff = (img1 - img2)/2
    avgdiff = np.mean(diff)
    print(avgdiff)
    return diff - avgdiff + 128

def coordinates_to_pixels(textfilepath): # From data_description.pdf, converts coordinates of ground truths to pixels.
    north_max = 7370488
    east_min = 1653166
    npoints = 25
    coord_mat = np.zeros((npoints, 2))
    pix_mat = np.zeros((npoints, 2))
    coord_mat[:, 0], coord_mat[:, 1] = np.loadtxt(textfilepath, delimiter = '\t', usecols = 0), np.loadtxt(textfilepath, delimiter = '\t', usecols = 1)
    #print(coord_mat.shape)
    pix_mat[:, 0] = np.round(north_max - coord_mat[:, 0])
    pix_mat[:, 1] = np.round(coord_mat[:, 1] - east_min)
    return

def standardize_pixels(Image): # Standardize (0 mean, 1 variance) image(s). 
    Image_norm = np.zeros(Image.shape)
    if len(Image.shape) == 3:
        for i in range(Image.shape[2]):
            mean = np.average(Image[:, :, i])
            std = np.std(Image[:, :, i]) + 1e-10
            Image_norm[:, :, i] = (Image[:, :, i] - mean)/std
    else:
        mean = np.average(Image)
        std = np.std(Image) + 1e-10
        Image_norm = (Image - mean)/std
    return Image_norm



# This function adds points around the ground truth centers (X = center of targets or nontargets, Y = class of target/nontarget -> 0 nontarget, 1 small target, 2 medium sized target, 3 big targets) 
# X.shape and Y.shape = (n_annotations, 2, n_images), assuming every image has the same number of annotated pixels for each kind
def insert_pixels_around_annotations(X, Y):  
  n_noisepoints = np.sum(Y[:, 1, 0] == 0)
  n_type1points = np.sum(Y[:, 1, 0] == 1)
  n_type2points = np.sum(Y[:, 1, 0] == 2)
  n_type3points = np.sum(Y[:, 1, 0] == 3)
  n_pixels_full = n_noisepoints*25 + n_type1points*9 + n_type2points*25 + n_type3points*25
  X_full = np.zeros((n_pixels_full, X.shape[1], X.shape[2]))
  Y_full = np.zeros((n_pixels_full, Y.shape[1], Y.shape[2]))
  for i in range(X.shape[2]): # For all examples
    k = 0
    for j in range(Y.shape[0]): # For all annotations
      if ((Y[j, 1, i] == 0) or (Y[j, 1, i] == 2) or (Y[j, 1, i] == 3)):
        X_full[k:(k+25), 0, i] = X[j, 0, i]*np.ones(25) + np.repeat(np.arange(-2, 3), 5)
        X_full[k:(k+25), 1, i] = X[j, 1, i]*np.ones(25) + np.tile(np.arange(-2, 3), 5) # Remember: (y,x). Following the horizontal axis (left to right) orientation [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)] 
        Y_full[k:(k+25), 0, i] = Y[j, 0, i]
        Y_full[k:(k+25), 1, i] = Y[j, 1, i]
        k += 25
      elif (Y[j, 1, i] == 1):
        X_full[k:(k+9), 0, i] = X[j, 0, i]*np.ones(9) + np.repeat(np.arange(-1, 2), 3)
        X_full[k:(k+9), 1, i] = X[j, 1, i]*np.ones(9) + np.tile(np.arange(-1, 2), 3) # Remember: (y,x). Following the horizontal axis (left to right) orientation [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)] 
        Y_full[k:(k+9), 0, i] = Y[j, 0, i]
        Y_full[k:(k+9), 1, i] = Y[j, 1, i]
        k += 9
      else:
        print("Error")
        break
    return X_full, Y_full




# This function retrieves the targets' centers of the specified difference image.
# img_name is the name of the desired difference image's name without the extension (.jpg)
def retrieve_xy_fromdifimg(img_name):
  tmp_mat = np.loadtxt(datab_imgs_path + "labels/" + img_name + ".txt", delimiter = ' ', usecols = range(4))
  X = tmp_mat[:, 0:2]
  Y = tmp_mat[:, 2:4]
  return X, Y


# This function creates a sparse matrix of the shape (3000, 2000) with ones in the positions where the targets are positioned. This is needed to train the pre-detection model.
def get_image_gt(img_names):
  Y_full = np.zeros( (len(img_names), im_dims[0], im_dims[1]) )
  for i, img_name in enumerate(img_names):
    X, Y = retrieve_xy_fromdifimg(img_name)
    X, Y = np.reshape(X, (-1, 2, 1)), np.reshape(Y, (-1, 2, 1)) 
    X, Y = insert_pixels_around_annotations(X, Y)
    X, Y = np.reshape(X, (-1, 2)), np.reshape(Y, (-1, 2)) 
    X = X[np.where(Y[:, 0] == 1)].astype(int)
    Y = np.zeros(im_dims)
    Y[X[:, 0], X[:, 1]] = 1
    Y = Y.astype(int)
    Y_full[i, :, :] = Y
  return Y_full


# This function retrieves the specified images, the standardized images and the binary ground truths for each image.
def get_images(whole_set):
  Images_vector = np.zeros( (len(whole_set), im_dims[0], im_dims[1] ) )
  for index, name in enumerate(whole_set):
    Images_vector[index, :, :] = bwimg(os.path.join(datab_imgs_path, name + ".jpg"))
  Images_vector_norm = standardize_pixels(Images_vector)
  Images_vector_norm = np.reshape(Images_vector_norm, (Images_vector_norm.shape[0], Images_vector_norm.shape[1], Images_vector_norm.shape[2], 1) )

  Y_full = get_image_gt(img_names)
  Y_full = np.reshape(Y_full, (Y_full.shape[0], Y_full.shape[1], Y_full.shape[2], 1) )

  return Images_vector, Images_vector_norm, Y_full



# Save model_conv/model_classconv:
def save_modelconv(modelvar, foldername, kfold):
  basepath = 'models/trainval/'
  modeldir = basepath + foldername + '/'
  try:
    os.mkdir(modeldir)
  except FileExistsError:
    pass  

  if kfold:
    for i, mod in enumerate(modelvar):
      mod.save(modeldir + "model_conv_" + str(i) + ".h5")
    with open(modeldir + 'summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
      modelvar[0].summary(print_fn=lambda x: fh.write(x + '\n'))
  else:
    modelvar.save(modeldir + "model_conv.h5")


# Load model_conv/model_classconv:
def load_modelconv(foldername, kfold):
  basepath = 'models/trainval/'
  modeldir = basepath + foldername + '/'
  floss = binary_focal_loss()

  if kfold:
    modelnames = ['model_conv_0.h5', 'model_conv_1.h5', 'model_conv_2.h5', 'model_conv_3.h5', 'model_conv_4.h5']
    n_split = len(modelnames)
    model_conv = np.empty(n_split, dtype=object)
    for i, modname in enumerate(modelnames):
      print(modname)
      model_conv[i] = load_model(modeldir + modname, custom_objects={'binary_focal_loss_fixed': floss })
  else:
    model_conv = load_model(modeldir + "model_conv.h5")
  
  return model_conv