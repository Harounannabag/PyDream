# imports and basic setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

import caffe

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
caffe.set_mode_gpu()
# select GPU device if multiple devices exist
caffe.set_device(0)

def showarray(a, filename, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO(filename)
    PIL.Image.fromarray(a).save(f.getvalue(), fmt)

# substitute your path here
model_path = './caffe/models/bvlc_googlenet/'
net_fn = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    # np.rollaxis rolls the specified axis to the start
    # [::-1] reverses the order of the axies from RGB to BGR
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    # first add back the mean
    # then change from BGR to RGB
    # note that img is of shape(3, h, w)
    # hence the dstack will stack each frame in the depth dim
    # yielding (h, w, 3)
    return np.dstack((img + net.transformer.mean['data'])[::-1])

# feel free to customize your objective
# here the L2 norm is used
# note that only the gradient is needed
# no actual loss is required
def objective_L2(dst):
    dst.diff[:] = dst.data 

def make_step(net, step_size = 1.5, end = 'inception_4c/output', 
              jitter = 32, clip = True, objective = objective_L2):
    '''Basic gradient ascent step.'''

    # input image is stored in Net's 'data' blob
    src = net.blobs['data']

    # end layer
    dst = net.blobs[end]

    # np.random.randint(low, height, size)
    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    
    # src.data is zeros((10, 3, 224, 224))
    # roll the last axis(x) backwards by ox
    # then roll the y axis backwards by oy
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
    
    # prepare the data and run the net until it reaches the *end*
    net.forward(end = end)
    
    # set the objective to be L2(1/2 * xi^2)
    # hence the difference(or delta) is just dst.data
    objective(dst)  # specify the optimization objective

    # propagate the gradient from the *end* layer
    net.backward(start = end)

    # get the diff at the src
    g = src.diff[0]

    # apply normalized ascent step to the input image(update happens here)
    src.data[:] += step_size/np.abs(g).mean() * g

    # unshift image
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)
            
    # note that the original image is meaned
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255 - bias)    

def deepdream(net, base_img, iter_n = 10, octave_n = 4, octave_scale = 1.4, 
              end = 'inception_4c/output', clip = True, **step_params):

    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        # zoom iteratively(from the largest to the smallest)
        # just another name for image reshapes
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order = 1))
    
    src = net.blobs['data']
    # allocate image for network-produced details
    detail = np.zeros_like(octaves[-1])

    # index and image
    # from the smallest to the largest
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # if not the orig image
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order = 1)

        # resize the network's input image size
        src.reshape(1, 3, h, w)

        # iteratively generate details
        src.data[0] = octave_base + detail

        # start training!
        for i in xrange(iter_n):
            make_step(net, end = end, clip = clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            # adjust image contrast if clipping is disabled
            if not clip:
                vis = vis * (255.0 / np.percentile(vis, 99.98))

            showarray(vis, "dream-octave{}-epoch{}.jpg".format(octave, i))
            print octave, i, end, vis.shape
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base

    # returning the resulting image
    return deprocess(net, src.data[0])

# substitute your image path here
img = np.float32(PIL.Image.open('image.jpg'))

_ = deepdream(net, img)

'''

The following code is adopted from the dream.ipynb.
The first part shows how a iteratively zooming in is applied
while the second part demonstartes how a new objective could be implemented.

# _ = deepdream(net, img, end = 'inception_3b/5x5_reduce')

# uncomment the next line to show all intermediate layers
# net.blobs.keys()

# frame generation

frame = img
frame_i = 0

h, w = frame.shape[:2]
# scale coefficient
s = 0.05
for i in xrange(100):
    frame = deepdream(net, frame)
    PIL.Image.fromarray(np.uint8(frame)).save("frames/%04d.jpg"%frame_i)
    frame = nd.affine_transform(frame, [1 - s, 1 - s, 1], [h * s / 2, w * s / 2, 0], order = 1)
    frame_i += 1

# guided generation
# substitute another image whose effect is desired to be inherited
guide = np.float32(PIL.Image.open('flowers.jpg'))

end = 'inception_3b/output'
h, w = guide.shape[:2]
src, dst = net.blobs['data'], net.blobs[end]
src.reshape(1, 3, h, w)
src.data[0] = preprocess(net, guide)
net.forward(end = end)
guide_features = dst.data[0].copy()

def objective_guide(dst):
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:, A.argmax(1)] # select ones that match best

_ = deepdream(net, img, end = end, objective = objective_guide)
'''
