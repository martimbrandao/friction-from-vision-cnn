#!/usr/bin/env python

import numpy as np
import scipy
import argparse
import cv2
import sys
import time

sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:
caffe_root = '/home/martim/workspace/cv/SegNet/caffe-segnet-cudnn5/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--colours', type=str, required=True)
args = parser.parse_args()

net = caffe.Net(args.model, args.weights, caffe.TEST)

# general params
strict_traversability = False

# quantile estimation params
Q_quantile = 0.05
Q_gpu = True # whether quantiles are computed in the network
Q_cpu = True # whether to compute quantiles on cpu
Q_search_tol = 0.025
Q_search_min = 0.0
Q_search_max = 0.8

# distribution parameters of material friction
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

# caffe does not let us change parameters on the power layer, so use what's on the prototxt
if Q_gpu:
  netproto = caffe_pb2.NetParameter()
  text_format.Merge(open(args.model).read(),netproto)
  Q_search_tol = netproto.layer[-1].power_param.scale
  Q_quantile = -netproto.layer[-4].power_param.shift
  print 'WARNING: Cannot change p nor tolerance parameters.'
  print '         Will use default from prototxt: Q_search_tol = ' + str(Q_search_tol)
  print '         Will use default from prototxt: Q_quantile = ' + str(Q_quantile)

# relax material friction parameters
if not strict_traversability:
  # set untraversable materials' avg such that q-quantile is zero friction
  # (less strict than enforcing avg=std=0)
  untraversable_friction_std = 0.10
  untraversable = material_friction_std == 0.0
  material_friction_std[untraversable] = untraversable_friction_std
  material_friction_avg[untraversable] = -scipy.stats.norm.ppf(
    Q_quantile, 0, untraversable_friction_std)

# custom weights for computing friction on network
if Q_gpu:
  n_materials = len(material_name)
  n_quantiles = net.params['cdf'][0].data.shape[0]
  for i in range(n_quantiles):
    p = Q_search_tol * i
    cdfs = scipy.stats.norm.cdf(p, material_friction_avg, material_friction_std)
    net.params['cdf'][0].data[i,:,0,0] = cdfs

# Proceed
caffe.set_mode_gpu()

input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape

label_colours = cv2.imread(args.colours).astype(np.uint8)

# Change this to your webcam ID, or file name for your video file
cap = cv2.VideoCapture(0)

if cap.isOpened(): # try to get the first frame
  rval, frame = cap.read()
else:
  rval = False

while rval:
  start = time.time()
  rval, frame = cap.read()
  end = time.time()
  print '%30s' % 'Grabbed camera frame in ', str((end - start)*1000), 'ms'

  start = time.time()
  frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
  input_image = frame.transpose((2,0,1))
  input_image = np.asarray([input_image])
  end = time.time()
  print '%30s' % 'Resized image in ', str((end - start)*1000), 'ms'

  start = time.time()
  out = net.forward_all(data=input_image)
  end = time.time()
  print '%30s' % 'Executed SegNet in ', str((end - start)*1000), 'ms'

  start = time.time()
  segmentation_ind = np.squeeze(net.blobs['argmax'].data)
  segmentation_ind_3ch = np.resize(segmentation_ind, (3,input_shape[2],input_shape[3]))
  segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
  segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

  cv2.LUT(segmentation_ind_3ch, label_colours, segmentation_rgb)
  segmentation_rgb = segmentation_rgb.astype(float)/255

  end = time.time()
  print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms'

  # martim (check for prediction "confidence" using the softmax value)
  start = time.time()
  predicted = net.blobs['prob'].data
  prob = np.squeeze(predicted[0,:,:,:])
  maxprob = np.max(prob, axis=0)
  segmentation_ind = np.squeeze(net.blobs['argmax'].data)
  #segmentation_ind[maxprob<0.95] = 255;

  segmentation_ind_3ch = np.resize(segmentation_ind, (3,input_shape[2],input_shape[3]))
  segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
  segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

  cv2.LUT(segmentation_ind_3ch, label_colours, segmentation_rgb)
  segmentation_rgb = segmentation_rgb.astype(float)/255

  end = time.time()
  print '%30s' % 'Processed results (v2) in ', str((end - start)*1000), 'ms'

  # compute quantile of friction for each pixel on the CPU
  # friction given label: f|l_k ~ N(mu_k,var_k)
  # friction given data:  p(f|data) = sum_k[p(f|l_k) * P(l_k|data)]
  # q-quantile of friction given data: Q(q) = F^-1(q) =
  #   {f : F(f|data) = q}, because F (cdf) strictly monotonically increasing
  if Q_cpu:
    # search the CDF for quantiles
    # F(f|data) = sum_k[F(f|l_k) * P(l_k|data)]
    start = time.time()
    F_cof_given_labels = lambda cof: scipy.stats.norm.cdf(
      cof,
      material_friction_avg,
      material_friction_std+np.array(1e-6)) # (m_labels,1)
    P_labels = np.reshape(
      prob,
      (prob.shape[0], prob.shape[1]*prob.shape[2])
      ).T # (n_pixels,m_labels)
    friction = Q_search_min * np.ones(P_labels.shape[0]) # (n_pixels,1)
    num_samples = 1 + (Q_search_max - Q_search_min) / Q_search_tol
    cofs = np.linspace(Q_search_min, Q_search_max, num_samples)
    # for each discretized quantile value
    for c in range(cofs.shape[0]):
      # as a matrix: F(f|data) = P(l_k|data) * F(f|l_k)
      F_cof = np.dot(P_labels, F_cof_given_labels(cofs[c])) # (n_pixels,1)
      # update friction quantiles by checking whether F<q
      friction[F_cof <= Q_quantile] = cofs[c]
    friction = np.reshape(friction.T, (prob.shape[1], prob.shape[2]))
    end = time.time()
    print '%30s' % 'Estimated friction in ', str((end - start)*1000), 'ms (on CPU)'

  # or just get the value from the network
  if Q_gpu:
    start = time.time()
    quantile_image = np.squeeze(net.blobs['Q'].data)
    end = time.time()
    print '%30s' % 'Got friction from cdf in ', str((end - start)*1000), 'ms (on GPU)\n'

  print ''

  cv2.imshow("Input", frame)
  cv2.imshow("Matrials", segmentation_rgb)
  if Q_cpu:
    cv2.imshow("FrictionQuantiles (CPU)", friction)
  if Q_gpu:
    cv2.imshow("FrictionQuantiles (GPU)", quantile_image)


  key = cv2.waitKey(1)
  if key == 27: # exit on ESC
    break
cap.release()
cv2.destroyAllWindows()

