import numpy as np
import sys
import caffe

caffe_dir = '/home/zhuo/caffe_python2/caffe/'

caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(caffe_dir+'models/vgg16/train_val.prototxt',
		caffe_dir+'models/vgg16/ssl_vggnet_train_iter_2600000_kernel_shape_decay_0.0001.caffemodel',
		caffe.TEST)

#print [(k, v.data.shape) for k, v in net.blobs.items()]
#print [(k, v[0].data.shape) for k, v in net.params.items()]
#print [(k, v[1].data.shape) for k, v in net.params.items()]

zerout_thres = 0.001
print 'zerout_thres: ', zerout_thres
for k,v in net.params.items():
	empty_channel_array = []
	empty_filter_array=[]
	conv_filter = v[0].data
	print conv_filter.shape
	print 'Filter Mean & std of layer {2}: {0}, {1}'.format(np.mean(np.absolute(conv_filter)), np.std(np.absolute(conv_filter)),k)
	if 'fc' in k:
		num_filter, num_channel = conv_filter.shape
		for i in range(num_filter):
			if np.mean(np.absolute(conv_filter[i,:])) < zerout_thres:
				empty_filter_array.append(i)
				#print 'fc filter {0} is empty'.format(i)
	else:
		num_filter, num_channel, height, width = conv_filter.shape
		for i in range(num_filter):
			empty_channel_in_filter=[]
			if np.mean(np.absolute(conv_filter[i,:,:,:])) < zerout_thres:
				#print 'conv filter {0} is empty'.format(i)
				empty_filter_array.append(i)
			for c in range(num_channel):
				if np.mean(np.absolute(conv_filter[i,c,:,:])) < zerout_thres:
					#print 'conv filter {0} channel {1} is empty'.format(i,c)
					empty_channel_in_filter.append(c)
			empty_channel_array.append(empty_channel_in_filter)
	result = set(empty_channel_array[0])
	#print result
	for s in empty_channel_array[1:]:
	    result.intersection_update(s)
	print 'empty filter %: {}'.format(len(empty_filter_array)/float(num_filter))
	print result
			

