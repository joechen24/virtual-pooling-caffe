from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

def gen_VIP_perlayer():
	net = caffe_pb2.NetParameter()

	fn = 'train_val_resnet50.prototxt'
	with open(fn) as f:
		s = f.read()
		txtf.Merge(s, net)

	layers = [l for l in net.layer]
	layerNames = [l.name for l in net.layer]
	reluLayers = [l for l in net.layer if l.type == 'ReLU']
	convLayers = [l for l in net.layer if l.type == 'Convolution']
	eltwiseLayers = [l for l in net.layer if l.type == 'Eltwise']
	eltwiseLayerNames = [l.name for l in net.layer if l.type == 'Eltwise']
	#print 'relu layers'
	#for reluLayer in reluLayers:
	#	print reluLayer.name
	#print 'Eltwise layers'
	#for eltwiseLayer in eltwiseLayers:
	#	#print eltwiseLayer.name
	#	for ind, bottomLayer in enumerate(eltwiseLayer.bottom):
	#		if bottomLayer not in eltwiseLayerNames:
	#			eltwiseLayer.bottom[ind] = 'interp/'+bottomLayer
	#			conv2change = bottomLayer
	#			convind = layerNames.index(conv2change)
	#			convlayer = net.layer[convind]
	#			print conv2change
	#			print convlayer.convolution_param.stride[0]
	count=0	
	interp_bottom_names = []
	interp_layer_names = []
	for convLayer in convLayers:
		interpLayerName = convLayer.name+'_relu'
		if interpLayerName not in layerNames:
			interpLayerName = 'scale'+convLayer.name.strip('res')
		#print '"{}",'.format(interpLayerName)
		interp_layer_names.append(interpLayerName)
		count+=1
		interp_bottom_name = convLayer.name
		print '"{}",'.format(convLayer.name)
		interp_bottom_names.append(interp_bottom_name)

	for interp_bottom_name, interpLayerName in zip(interp_bottom_names, interp_layer_names):
		net = caffe_pb2.NetParameter()
		fn = 'train_val_resnet50.prototxt'
		with open(fn) as f:
			s = f.read()
			txtf.Merge(s, net)

		layers = [l for l in net.layer]
		layerNames = [l.name for l in net.layer]
		tmp_ind = layerNames.index(interp_bottom_name)
		convLayer = net.layer[tmp_ind]
		convLayer.convolution_param.stride[0] = 2*convLayer.convolution_param.stride[0]
		for layeridx, layer in enumerate(layers):
			if interp_bottom_name in layer.bottom and layeridx > layerNames.index(interpLayerName):
				for ind, bottomLayer in enumerate(layer.bottom):
					if bottomLayer == interp_bottom_name:
						interp_top_name = 'lininterp/'+interp_bottom_name
						layers[layeridx].bottom[ind] = interp_top_name
				outFn = './tmp_train_val/tmp_train_val_resnet50_{}.prototxt'.format(interpLayerName)
				#print 'writing', outFn
				with open(outFn, 'w') as f:
					f.write(str(net))

				newFn = './tmp_train_val/train_val_resnet50_{}.prototxt'.format(interpLayerName)
				with open(outFn) as f:
					with open(newFn,'w') as newf:
						flag = 0
						for line in f.readlines():
							if interpLayerName in line:
								# accommondate the last few layers with 7*7 size
								if '5a' in interpLayerName or '5b' in interpLayerName or '5c' in interpLayerName:
									flag = 2
								else:
									flag=1
							if flag == 2 and 'layer' in line:
								flag = 0
								newf.write('layer {\n')
								newf.write('  type: "Interp"\n')
								newf.write('  name: "{}"\n'.format(interp_top_name+'_tmp'))
								newf.write('  top: "{}"\n'.format(interp_top_name+'_tmp'))
								newf.write('  bottom: "{}"\n'.format(interp_bottom_name))
								newf.write('}\n')

								newf.write('layer {\n')
								newf.write('  type: "DummyData"\n')
								newf.write('  name: "{}"\n'.format(interp_top_name+'_dummy'))
								newf.write('  top: "{}"\n'.format(interp_top_name+'_dummy'))
								newf.write('  dummy_data_param{\n')
								newf.write('    num: 50\n')
								newf.write('    channels: 512\n')
								newf.write('    width: 7\n')
								newf.write('    height: 7\n')
								newf.write('    }\n')
								newf.write('}\n')

								newf.write('layer {\n')
								newf.write('  type: "Crop"\n')
								newf.write('  name: "{}"\n'.format(interp_top_name))
								newf.write('  bottom: "{}"\n'.format(interp_top_name+'_tmp'))
								newf.write('  bottom: "{}"\n'.format(interp_top_name+'_dummy'))
								newf.write('  top: "{}"\n'.format(interp_top_name))
								newf.write('  crop_param{\n')
								newf.write('    axis: 2\n')
								newf.write('    offset: 0\n')
								newf.write('    }\n')
								newf.write('}\n')
							if flag == 1 and 'layer' in line:
								flag = 0
								newf.write('layer {\n')
								newf.write('  type: "Interp"\n')
								newf.write('  name: "{}"\n'.format(interp_top_name))
								newf.write('  top: "{}"\n'.format(interp_top_name))
								newf.write('  bottom: "{}"\n'.format(interp_bottom_name))
								newf.write('}\n')
							newf.write(line)
				       		
				print "./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_{0}.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_{0}.out".format(interpLayerName)
				break  # should I break here? what if multiple ones using the output of this interpolation?

		
		
	#idx = layerNames.index('fc6')
	#l = net.layer[idx]
	#l.param[0].lr_mult = 1.3

	#outFn = 'mod_train_val_test.prototxt'
	#print 'writing', outFn
	#with open(outFn, 'w') as f:
	#	f.write(str(net))


def gen_VIP_perlayer_new():
	net = caffe_pb2.NetParameter()

	fn = 'train_val_resnet50.prototxt'
	with open(fn) as f:
		s = f.read()
		txtf.Merge(s, net)

	layers = [l for l in net.layer]
	layerNames = [l.name for l in net.layer]
	reluLayers = [l for l in net.layer if l.type == 'ReLU']
	convLayers = [l for l in net.layer if l.type == 'Convolution']
	eltwiseLayers = [l for l in net.layer if l.type == 'Eltwise']
	eltwiseLayerNames = [l.name for l in net.layer if l.type == 'Eltwise']
	count=0	
	interp_bottom_names = []
	interp_layer_names = []
	for convLayer in convLayers:
		interpLayerName = convLayer.name+'_relu'
		if interpLayerName not in layerNames:
			interpLayerName = 'scale'+convLayer.name.strip('res')
		#print '"{}",'.format(interpLayerName)
		interp_layer_names.append(interpLayerName)
		count+=1
		interp_bottom_name = convLayer.name
		print '"{}",'.format(convLayer.name)
		interp_bottom_names.append(interp_bottom_name)

	for interp_bottom_name, interpLayerName in zip(interp_bottom_names, interp_layer_names):
		net = caffe_pb2.NetParameter()
		fn = 'train_val_resnet50.prototxt'
		with open(fn) as f:
			s = f.read()
			txtf.Merge(s, net)

		layers = [l for l in net.layer]
		layerNames = [l.name for l in net.layer]
		tmp_ind = layerNames.index(interp_bottom_name)
		convLayer = net.layer[tmp_ind]
		convLayer.convolution_param.stride[0] = 2*convLayer.convolution_param.stride[0]
		for layeridx, layer in enumerate(layers):
			if interp_bottom_name in layer.bottom and layeridx > layerNames.index(interpLayerName):
				for ind, bottomLayer in enumerate(layer.bottom):
					if bottomLayer == interp_bottom_name:
						interp_top_name = 'lininterp/'+interp_bottom_name
						layers[layeridx].bottom[ind] = interp_top_name
				outFn = './tmp_train_val/tmp_train_val_resnet50_{}.prototxt'.format(interpLayerName)
				#print 'writing', outFn
				with open(outFn, 'w') as f:
					f.write(str(net))

				newFn = './tmp_train_val/train_val_resnet50_{}.prototxt'.format(interpLayerName)
				with open(outFn) as f:
					with open(newFn,'w') as newf:
						flag = 0
						for line in f.readlines():
							if interpLayerName in line:
								flag=1
							if flag == 1 and 'layer' in line:
								flag = 0
								newf.write('layer {\n')
								newf.write('  type: "Interp"\n')
								newf.write('  name: "{}"\n'.format(interp_top_name))
								newf.write('  top: "{}"\n'.format(interp_top_name))
								newf.write('  bottom: "{}"\n'.format(interp_bottom_name))
								newf.write('}\n')
							newf.write(line)
				       		
				print "./build/tools/caffe test --model=models/resnet/tmp_train_val/train_val_resnet50_{0}.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 -iterations 1000 2>&1 | tee ./resnet_results/resnet50_lininterp_{0}.out".format(interpLayerName)
				break  # should I break here? what if multiple ones using the output of this interpolation?

def gen_VIP_round(roundInd,layers_to_interp):
	net = caffe_pb2.NetParameter()

	fn = 'train_val_resnet50.prototxt'
	with open(fn) as f:
		s = f.read()
		txtf.Merge(s, net)

	layers = [l for l in net.layer]
	layerNames = [l.name for l in net.layer]
	convLayers = [l for l in net.layer if l.type == 'Convolution']
	#convLayerNames = [l.name for l in net.layer if l.type == 'Convolution']

	count=0	
	interp_bottom_names = []
	interp_top_names = []
	interp_layer_names = []
	for convLayer in convLayers:
		if convLayer.name in layers_to_interp:
			convLayer.convolution_param.stride[0] = 2*convLayer.convolution_param.stride[0]
			interpLayerName = convLayer.name+'_relu'
			if interpLayerName not in layerNames:
				interpLayerName = 'scale'+convLayer.name.strip('res')
			print '"{}",'.format(interpLayerName)
			interp_layer_names.append(interpLayerName)
			count+=1
			interp_bottom_name = convLayer.name
			interp_bottom_names.append(interp_bottom_name)

	for interp_bottom_name, interpLayerName in zip(interp_bottom_names, interp_layer_names):
		for layeridx, layer in enumerate(layers):
			if interp_bottom_name in layer.bottom and layeridx > layerNames.index(interpLayerName):
				for ind, bottomLayer in enumerate(layer.bottom):
					if bottomLayer == interp_bottom_name:
						interp_top_name = 'lininterp/'+interp_bottom_name
						interp_top_names.append(interp_top_name)
						layers[layeridx].bottom[ind] = interp_top_name
				#break   ## no break here
	outFn = './tmp_train_val/tmp_train_val_resnet50_round{}.prototxt'.format(roundInd)
	#print 'writing', outFn
	with open(outFn, 'w') as f:
		f.write(str(net))

	newFn = './tmp_train_val/train_val_resnet50_round{}.prototxt'.format(roundInd)
	with open(outFn) as f:
		with open(newFn,'w') as newf:
			flag = 0
			dummyExist = 0
			for line in f.readlines():
				for interpIdx, interpLayerName in enumerate(interp_layer_names):
					if interpLayerName in line:
						if '5a' in interpLayerName or '5b' in interpLayerName or '5c' in interpLayerName:
							flag = 2
						else:
							flag=1
						interp_top_name = interp_top_names[interpIdx]
						interp_bottom_name = interp_bottom_names[interpIdx]
						break
					if flag == 2 and 'layer' in line:
						flag = 0
						if not dummyExist:
							newf.write('layer {\n')
							newf.write('  type: "DummyData"\n')
							newf.write('  name: "{}"\n'.format('onlyone_dummy'))
							newf.write('  top: "{}"\n'.format('onlyone_dummy'))
							newf.write('  dummy_data_param{\n')
							newf.write('    num: 50\n')
							newf.write('    channels: 512\n')
							newf.write('    width: 7\n')
							newf.write('    height: 7\n')
							newf.write('    }\n')
							newf.write('}\n')
							dummyExist = 1
						newf.write('layer {\n')
						newf.write('  type: "Interp"\n')
						newf.write('  name: "{}"\n'.format(interp_top_name+'_tmp'))
						newf.write('  top: "{}"\n'.format(interp_top_name+'_tmp'))
						newf.write('  bottom: "{}"\n'.format(interp_bottom_name))
						newf.write('}\n')


						newf.write('layer {\n')
						newf.write('  type: "Crop"\n')
						newf.write('  name: "{}"\n'.format(interp_top_name))
						newf.write('  bottom: "{}"\n'.format(interp_top_name+'_tmp'))
						newf.write('  bottom: "{}"\n'.format('onlyone_dummy'))
						newf.write('  top: "{}"\n'.format(interp_top_name))
						newf.write('  crop_param{\n')
						newf.write('    axis: 2\n')
						newf.write('    offset: 0\n')
						newf.write('    }\n')
						newf.write('}\n')
					if flag == 1 and 'layer' in line:
						flag = 0
						newf.write('layer {\n')
						newf.write('  type: "Interp"\n')
						newf.write('  name: "{}"\n'.format(interp_top_name))
						newf.write('  top: "{}"\n'.format(interp_top_name))
						newf.write('  bottom: "{}"\n'.format(interp_bottom_name))
						newf.write('}\n')
						break
					elif flag==1:
						break
				newf.write(line)
				       		
	print './build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round{0}.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round{0}_interpCUDAv3.out'.format(roundInd)
	print "./build/tools/caffe train --solver=models/resnet/solver_resnet50_round{0}.prototxt --weights=models/resnet/ResNet-50-model.caffemodel -gpu 0 2>&1 | tee ./resnet_results/resnet50_lininterp_round{0}.out".format(roundInd)


def gen_VIP_round_new(roundInd,layers_to_interp):
	net = caffe_pb2.NetParameter()

	fn = 'train_val_resnet50_train33test1.prototxt'
	with open(fn) as f:
		s = f.read()
		txtf.Merge(s, net)

	layers = [l for l in net.layer]
	layerNames = [l.name for l in net.layer]
	convLayers = [l for l in net.layer if l.type == 'Convolution']
	#convLayerNames = [l.name for l in net.layer if l.type == 'Convolution']

	count=0	
	interp_bottom_names = []
	interp_top_names = []
	interp_layer_names = []
	for convLayer in convLayers:
		if convLayer.name in layers_to_interp:
			convLayer.convolution_param.stride[0] = 2*convLayer.convolution_param.stride[0]
			interpLayerName = convLayer.name+'_relu'
			if interpLayerName not in layerNames:
				interpLayerName = 'scale'+convLayer.name.strip('res')
			#print '"{}",'.format(interpLayerName)
			interp_layer_names.append(interpLayerName)
			count+=1
			interp_bottom_name = convLayer.name
			interp_bottom_names.append(interp_bottom_name)

	for interp_bottom_name, interpLayerName in zip(interp_bottom_names, interp_layer_names):
		for layeridx, layer in enumerate(layers):
			if interp_bottom_name in layer.bottom and layeridx > layerNames.index(interpLayerName):
				for ind, bottomLayer in enumerate(layer.bottom):
					if bottomLayer == interp_bottom_name:
						interp_top_name = 'lininterp/'+interp_bottom_name
						interp_top_names.append(interp_top_name)
						layers[layeridx].bottom[ind] = interp_top_name
				#break   ## no break here
	outFn = './tmp_train_val/tmp_train_val_resnet50_round{}_train33test1.prototxt'.format(roundInd)
	#print 'writing', outFn
	with open(outFn, 'w') as f:
		f.write(str(net))

	newFn = './tmp_train_val/train_val_resnet50_round{}_train33test1.prototxt'.format(roundInd)
	with open(outFn) as f:
		with open(newFn,'w') as newf:
			flag = 0
			dummyExist = 0
			for line in f.readlines():
				for interpIdx, interpLayerName in enumerate(interp_layer_names):
					if interpLayerName in line:
						flag=1
						interp_top_name = interp_top_names[interpIdx]
						interp_bottom_name = interp_bottom_names[interpIdx]
						break
					if flag == 1 and 'layer' in line:
						flag = 0
						newf.write('layer {\n')
						newf.write('  type: "Interp"\n')
						newf.write('  name: "{}"\n'.format(interp_top_name))
						newf.write('  top: "{}"\n'.format(interp_top_name))
						newf.write('  bottom: "{}"\n'.format(interp_bottom_name))
						newf.write('}\n')
						break
					elif flag==1:
						break
				newf.write(line)
				       		
	#solverTemplateFn = './tmp_solver/solver_resnet50_template.prototxt'
	#solverFn = './tmp_solver/solver_resnet50_round{}.prototxt'.format(roundInd)
	#with open(solverTemplateFn) as tempf:
	#	with open(solverFn, 'w') as f:
	#		f.write('net: "models/resnet/tmp_train_val/train_val_resnet50_round{}.prototxt"\n'.format(roundInd))
	#		for line in tempf.readlines():
	#			f.write(line)
	#		f.write('snapshot_prefix: "models/resnet/snapshots/resnet50_lininterp_finetune{}_CUDAv5_newLRpolicyFrom1e-4_35batchsize"'.format(roundInd))
			
	print './build/tools/caffe time --model=models/resnet/tmp_train_val/train_val_resnet50_round{0}_train33test1.prototxt  -gpu 1  2>&1 | tee  ./resnet_results/time_resnet50_lininterp_round{0}_interpCUDAv5_train33test1.out'.format(roundInd)
	#print "./build/tools/caffe train --solver=models/resnet/tmp_solver/solver_resnet50_round{0}.prototxt --weights=models/resnet/snapshots/resnet50_lininterp_finetune{1}_CUDAv5_newLRpolicy_35batchsize.caffemodel -gpu 1 2>&1 | tee ./resnet_results/resnet50_lininterp_round{0}_CUDAv5_newLRpolicy_35batchsize.out".format(roundInd, roundInd-1)

########################################========= main ===========###############################
all_layers_to_interp=[
	"origin",
	"conv1",
	"res2a_branch1",
	"res2a_branch2a",
	"res2a_branch2b",
	"res2a_branch2c",
	"res2b_branch2a",
	"res2b_branch2b",
	"res2b_branch2c",
	"res2c_branch2a",
	"res2c_branch2b",
	"res2c_branch2c",
	"res3a_branch1",
	"res3a_branch2a",
	"res3a_branch2b",
	"res3a_branch2c",
	"res3b_branch2a",
	"res3b_branch2b",
	"res3b_branch2c",
	"res3c_branch2a",
	"res3c_branch2b",
	"res3c_branch2c",
	"res3d_branch2a",
	"res3d_branch2b",
	"res3d_branch2c",
	"res4a_branch1",
	"res4a_branch2a",
	"res4a_branch2b",
	"res4a_branch2c",
	"res4b_branch2a",
	"res4b_branch2b",
	"res4b_branch2c",
	"res4c_branch2a",
	"res4c_branch2b",
	"res4c_branch2c",
	"res4d_branch2a",
	"res4d_branch2b",
	"res4d_branch2c",
	"res4e_branch2a",
	"res4e_branch2b",
	"res4e_branch2c",
	"res4f_branch2a",
	"res4f_branch2b",
	"res4f_branch2c",
	"res5a_branch1",
	"res5a_branch2a",
	"res5a_branch2b",
	"res5a_branch2c",
	"res5b_branch2a",
	"res5b_branch2b",
	"res5b_branch2c",
	"res5c_branch2a",
	"res5c_branch2b",
	"res5c_branch2c"
]
roundInd = 3
round3Elem = [27,28,35,45,4,5,26,1,14,15,3,41,51,22]
if roundInd == 1:
	###### for round 1
	layers_to_interp = [all_layers_to_interp[i] for i in [10,11,17,18,19,20,23,24,29,30,31,32,33,34,36,37,38,39,40,42,43]]
elif roundInd == 2:
	###### for round 2
	layers_to_interp = [all_layers_to_interp[i] for i in [10,11,17,18,19,20,23,24,29,30,31,32,33,34,36,37,38,39,40,42,43,21,46,47,48,49,50,52,53]]
elif roundInd == 3:
	###### for round 3
	for perlayer in range(len(round3Elem)):
		layers_to_interp = [all_layers_to_interp[i] for i in [10,11,17,18,19,20,23,24,29,30,31,32,33,34,36,37,38,39,40,42,43,21,46,47,48,49,50,52,53]+round3Elem[:perlayer+1]]
		gen_VIP_round_new(roundInd*10+perlayer+1,layers_to_interp)
#gen_VIP_perlayer_new()
