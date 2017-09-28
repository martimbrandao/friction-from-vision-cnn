#!/usr/bin/env python

# NOTES:
# - for some reason, on my computer caffe-segnet-cudnn5 runs very slow on the rosnode (but not on demo)
# - use original caffe-segnet here if you have the same problem. just compile it without cuDNN,
#   it is not that slow

# general
import sys, time
import numpy as np
from scipy.stats import norm

# cv
import cv2

# segnet
import os.path
import scipy
sys.path.append('/usr/local/lib/python2.7/site-packages')
caffe_root = '/home/martim/workspace/cv/SegNet/caffe-segnet-cudnn5/' # use the origin caffe-segnet if this one is slow
sys.path.insert(0, caffe_root + 'python')
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

# ros
import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from friction_from_vision.srv import EstimateMaterialsAndFriction,EstimateMaterialsAndFrictionResponse

################################################################################

### general params
fast_publishing = True
bayesian = False
hardcoded_colors = True
save_to_file = False
strict_traversability = False # False will lower the influence of
                              # untraversable materials on friction quantiles

### quantile estimation params
Q_quantile = 0.05 # the p in p-quantile
Q_gpu = True # whether quantiles are computed in the network
Q_method = 'search' # 'search' or 'approx'. search is better but slower
Q_search_tol = 0.025 # this is the discretization used during quantile search
Q_search_min = 0.0
Q_search_max = 0.8

### humanoids2016 table (note avg=std=0 for untraversable can be too strict)
material_name = [
  "Asphalt",
  "Grass",
  "Rock",
  "Sand",
  "Sky",
  "Snow",
  "Water",
  "Carpet",
  "Ceramic",
  "Cloth",
  "Marble",
  "Metal",
  "Paper",
  "Wood"
]
material_friction_avg = np.array([
  0.74, # asphalt
  0.53, # grass
  0.80, # rock
  0.00, # sand
  0.00, # sky
  0.00, # snow
  0.00, # water
  0.82, # carpet
  0.97, # ceramic
  0.00, # cloth
  0.83, # marble
  0.80, # metal
  0.00, # paper
  0.88  # wood
])
material_friction_std = np.array([
  0.12, # asphalt
  0.10, # grass
  0.08, # rock
  0.00, # sand
  0.00, # sky
  0.00, # snow
  0.00, # water
  0.02, # carpet
  0.05, # ceramic
  0.00, # cloth
  0.15, # marble
  0.15, # metal
  0.00, # paper
  0.12  # wood
])

### label colors
if hardcoded_colors:
  c0 = [77,175,74]
  c1 = [228,26,28]
  c2 = [55,126,184]
  c3 = [152,78,163]
  c4 = [255,127,0]
  c5 = [255,255,51]
  c6 = [166,86,40]
  c7 = [247,129,191]
  c8 = [153,153,153]
  c9 = [0,0,255]
  c10 = [255,0,255]
  c11 = [0,255,248]
  c12 = [0,255,0]
  c13 = [0,101,255]
  c14 = [255,255,180]
  label_colours = np.array([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14])

################################################################################

def cv2ros(img):
  try:
    sys.stdout = open(os.devnull, "w") #hack because ros is printing some stuff inside...
    msg = bridge.cv2_to_imgmsg(img, encoding="passthrough") #convert
    sys.stdout = sys.__stdout__ #hack end
  except CvBridgeError as e:
    print e
  return msg

################################################################################

def processImage(rosImg):

  # convert from ros to opencv
  start = time.time()
  if isinstance(rosImg, CompressedImage):
    # compressed: direct conversion to CV2
    np_arr = np.fromstring(rosImg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
  else:
    # image: cv bridge
    try:
      image_np = bridge.imgmsg_to_cv2(rosImg, "bgr8")
    except CvBridgeError as e:
      print e

  end = time.time()
  print '%30s' % 'Converted image in ', str((end - start)*1000), 'ms'

  # resize
  start = time.time()
  frame = cv2.resize(image_np, (input_shape[3],input_shape[2]))
  input_image = frame.transpose((2,0,1))
  input_image = np.asarray([input_image])
  if bayesian:
    input_image = np.repeat(input_image,input_shape[0],axis=0)

  end = time.time()
  print '%30s' % 'Resized image in ', str((end - start)*1000), 'ms'

  # forward pass on CNN using SegNet
  start = time.time()
  out = net.forward_all(data=input_image)
  end = time.time()
  print '%30s' % 'Forward pass in ', str((end - start)*1000), 'ms'

  # get segmentation
  start = time.time()
  predicted = net.blobs['prob'].data
  if bayesian:
    # pdf and max segmentation
    prob = np.mean(predicted,axis=0)
    segmentation_ind = np.argmax(prob, axis=0).astype(np.uint8)
    uncertainty = np.var(predicted,axis=0)
    avg_uncertainty = np.mean(uncertainty,axis=0)
    # model uncertainty threshold
    avg_ = np.mean(avg_uncertainty)
    std_ = np.std(avg_uncertainty)
    segmentation_ind[avg_uncertainty>avg_+std_] = 14
    # class uncertainty threshold
    maxprob = np.max(prob, axis=0)
    segmentation_ind[maxprob<0.33] = 14
  else:
    # pdf and max segmentation
    prob = np.squeeze(predicted[0,:,:,:])
    segmentation_ind = np.squeeze(net.blobs['argmax'].data).astype(np.uint8)
    # class uncertainty threshold
    maxprob = np.max(prob, axis=0)
    segmentation_ind[maxprob<0.33] = 14

  end = time.time()
  print '%30s' % 'Got segmentation in ', str((end - start)*1000), 'ms'

  # compute quantile of friction for each pixel
  # friction given label: f|l_k ~ N(mu_k,var_k)
  # friction given data:  p(f|data) = sum_k[p(f|l_k) * P(l_k|data)]
  # q-quantile of friction given data:  Q(q) = F^-1(q) =
  #   {f : F(f|data) = q}, because F strictly monotonically increasing
  if Q_method == 'search':
    # option (1): by searching the CDF
    # F(f|data) = sum_k[F(f|l_k) * P(l_k|data)]
    start = time.time()
    if 'Q' in net.blobs:
      # just get the result from the network
      friction = np.squeeze(net.blobs['Q'].data)
    else:
      # do the search on CPU
      F_cof_given_labels = lambda cof: norm.cdf(
        cof,
        material_friction_avg,
        material_friction_std+np.array(1e-6)) # (m_labels,1)
      P_labels = np.reshape(
        prob,
        (prob.shape[0],prob.shape[1]*prob.shape[2])
        ).T # (n_pixels,m_labels)
      friction = Q_search_min * np.ones(P_labels.shape[0]) # (n_pixels,1)
      num_samples = 1 + (Q_search_max - Q_search_min) / Q_search_tol
      cofs = np.linspace(Q_search_min, Q_search_max, num_samples)
      for c in range(cofs.shape[0]):
        # as a matrix: F(f|data) = P(l_k|data) * F(f|l_k)
        F_cof = np.dot(P_labels, F_cof_given_labels(cofs[c])) # (n_pixels,1)
        # update friction quantiles by checking whether F<q
        friction[F_cof <= Q_quantile] = cofs[c]
      friction = np.reshape(friction.T, (prob.shape[1],prob.shape[2]))
    end = time.time()
    print '%30s' % 'Estimated friction in ', str((end - start)*1000), 'ms'
  else:
    # option (2): by approximating by a single Gaussian (RMSE = 0.1)
    # mu  = sum [P(l_k|data) * mu_k]
    # var = sum [P(l_k|data) * (mu_k^2 + var_k)] - mu^2
    start = time.time()
    friction_gaussian_avg = np.sum(prob * material_friction_avg_matrix, axis=0)
    friction_gaussian_var = np.sum(prob * material_friction_avg2_std2_matrix, axis=0) - np.square(friction_gaussian_avg)
    friction = friction_gaussian_avg + norm.ppf(Q_quantile)*np.sqrt(friction_gaussian_var)
    friction[friction < 0] = 0
    end = time.time()
    print '%30s' % 'Estimated approx friction in ', str((end - start)*1000), 'ms'

  # rgb image
  if not fast_publishing:
    segmentation_ind_3ch = np.resize(segmentation_ind, (3,input_shape[2],input_shape[3]))
    segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
    segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)
    cv2.LUT(segmentation_ind_3ch, label_colours, segmentation_rgb)
    segmentation_rgb = segmentation_rgb.astype(float)/255
  else:
    segmentation_rgb = None

  # save
  if save_to_file:
    friction2 = friction
    friction2[friction2 < 0] = 0
    cv2.imwrite(mypath+'/'+str(rosImg.header.seq)+'_img.png', frame)
    cv2.imwrite(mypath+'/'+str(rosImg.header.seq)+'_pred.png', segmentation_rgb*255)
    cv2.imwrite(mypath+'/'+str(rosImg.header.seq)+'_cof.png', (1-friction2)*255)

  # show
  if not fast_publishing:
    friction2 = friction
    friction2[friction2 < 0] = 0
    cv2.imshow("SegNet input", frame)
    cv2.imshow("SegNet materials", segmentation_rgb)
    cv2.imshow("SegNet friction", 1-friction2)
    cv2.waitKey(1)

  # return
  print ''
  return segmentation_ind, segmentation_rgb, friction

################################################################################

def callback(rosImg):

  # process image
  segmentation_ind, segmentation_rgb, friction = processImage(rosImg)

  # publish materials
  msg = cv2ros(segmentation_ind)
  msg.header.stamp = rosImg.header.stamp
  pub_materials.publish(msg)

  # publish friction
  msg = cv2ros(friction)
  msg.header.stamp = rosImg.header.stamp
  pub_friction.publish(msg)

  # publish materials RGB
  if not fast_publishing:
    msg = cv2ros(segmentation_rgb)
    msg.header.stamp = rosImg.header.stamp
    pub_materials_rgb.publish(msg)

################################################################################

def service(req):

  # process image
  segmentation_ind, segmentation_rgb, friction = processImage(req.image)

  res = EstimateMaterialsAndFrictionResponse()
  res.materials = cv2ros(segmentation_ind)
  res.friction = cv2ros(friction)
  return res

################################################################################

if __name__ == '__main__':

  # ros
  rospy.init_node('friction_from_vision', anonymous=True)
  bridge = CvBridge()

  # ros topics
  subscriber = rospy.Subscriber("/multisense/left/image_rect_color", Image, callback, queue_size=1)

  # ros publisher
  pub_materials = rospy.Publisher("/wabian/image_materials", Image, queue_size=10)
  pub_materials_rgb = rospy.Publisher("/wabian/image_materials_rgb", Image, queue_size=10)
  pub_friction = rospy.Publisher("/wabian/image_friction", Image, queue_size=10)

  # ros service
  srv = rospy.Service('/wabian/estimate_materials_and_friction', EstimateMaterialsAndFriction, service)

  # segnet config
  rospack = rospkg.RosPack()
  mypath = rospack.get_path('friction_from_vision')
  if Q_gpu:
    # friction quantiles computed on the network
    arg_model = mypath+"/data/segnet_webdemo2_friction.prototxt"
    arg_weights = mypath+"/data/segnet_test_weights2.caffemodel"
  else:
    if bayesian:
      arg_model = mypath+"/data/bayesian_segnet_inference2.prototxt"
      arg_weights = mypath+"/data/bayesian_segnet_test_weights.caffemodel"
    else:
      arg_model = mypath+"/data/segnet_webdemo2.prototxt"
      arg_weights = mypath+"/data/segnet_test_weights2.caffemodel"
  arg_colours = mypath+"/data/mybigdataset2.png"
  net = caffe.Net(arg_model, arg_weights, caffe.TEST)
  caffe.set_mode_gpu()

  input_shape = net.blobs['data'].data.shape
  print "input_shape:"
  print input_shape

  # caffe does not let us change parameters on the power layer, so use what's on the prototxt
  if Q_gpu:
    netproto = caffe_pb2.NetParameter()
    text_format.Merge(open(arg_model).read(),netproto)
    Q_search_tol = netproto.layer[-1].power_param.scale
    Q_quantile = -netproto.layer[-4].power_param.shift
    print 'WARNING: Cannot change p nor tolerance parameters.'
    print '         Will use default from prototxt: Q_search_tol = ' + str(Q_search_tol)
    print '         Will use default from prototxt: Q_quantile = ' + str(Q_quantile)

  # relax material friction parameters
  if not strict_traversability:
    # set untraversable materials' avg such that p-quantile is zero friction
    # (less strict than enforcing avg=std=0)
    untraversable_friction_std = 0.10
    untraversable = material_friction_std == 0.0
    material_friction_std[untraversable] = untraversable_friction_std
    material_friction_avg[untraversable] = -norm.ppf(Q_quantile, 0, untraversable_friction_std)

  # utility matrices
  material_friction_avg_matrix = np.ones((14,360,480))
  material_friction_std_matrix = np.ones((14,360,480))
  material_friction_var_matrix = np.ones((14,360,480))
  material_friction_avg2_std2_matrix = np.ones((14,360,480))
  for k in range(material_friction_avg_matrix.shape[0]):
    for i in range(material_friction_avg_matrix.shape[1]):
      for j in range(material_friction_avg_matrix.shape[2]):
        material_friction_avg_matrix[k,i,j] = material_friction_avg[k]
        material_friction_std_matrix[k,i,j] = material_friction_std[k]
        material_friction_var_matrix[k,i,j] = np.square(material_friction_std[k])
        material_friction_avg2_std2_matrix[k,i,j] = np.square(material_friction_avg[k]) + np.square(material_friction_std[k])

  # label colours
  colours = cv2.imread(arg_colours).astype(np.uint8)
  if hardcoded_colors:
    for i in range(label_colours.shape[0]):
      colours[0,i,:] = label_colours[i,::-1]
  label_colours = colours

  # custom weights for computing friction on network
  if Q_gpu:
    n_materials = len(material_name)
    n_quantiles = net.params['cdf'][0].data.shape[0]
    for i in range(n_quantiles):
      p = Q_search_tol * i
      cdfs = scipy.stats.norm.cdf(p, material_friction_avg, material_friction_std)
      net.params['cdf'][0].data[i,:,0,0] = cdfs

  # ros eternal loop
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down"
  cv2.destroyAllWindows()

