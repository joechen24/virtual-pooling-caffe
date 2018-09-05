def parse_perlayer():
	files = [
		"origin",
		"conv1_relu",
		"scale2a_branch1",
		"res2a_branch2a_relu",
		"res2a_branch2b_relu",
		"scale2a_branch2c",
		"res2b_branch2a_relu",
		"res2b_branch2b_relu",
		"scale2b_branch2c",
		"res2c_branch2a_relu",
		"res2c_branch2b_relu",
		"scale2c_branch2c",
		"scale3a_branch1",
		"res3a_branch2a_relu",
		"res3a_branch2b_relu",
		"scale3a_branch2c",
		"res3b_branch2a_relu",
		"res3b_branch2b_relu",
		"scale3b_branch2c",
		"res3c_branch2a_relu",
		"res3c_branch2b_relu",
		"scale3c_branch2c",
		"res3d_branch2a_relu",
		"res3d_branch2b_relu",
		"scale3d_branch2c",
		"scale4a_branch1",
		"res4a_branch2a_relu",
		"res4a_branch2b_relu",
		"scale4a_branch2c",
		"res4b_branch2a_relu",
		"res4b_branch2b_relu",
		"scale4b_branch2c",
		"res4c_branch2a_relu",
		"res4c_branch2b_relu",
		"scale4c_branch2c",
		"res4d_branch2a_relu",
		"res4d_branch2b_relu",
		"scale4d_branch2c",
		"res4e_branch2a_relu",
		"res4e_branch2b_relu",
		"scale4e_branch2c",
		"res4f_branch2a_relu",
		"res4f_branch2b_relu",
		"scale4f_branch2c",
		"scale5a_branch1",
		"res5a_branch2a_relu",
		"res5a_branch2b_relu",
		"scale5a_branch2c",
		"res5b_branch2a_relu",
		"res5b_branch2b_relu",
		"scale5b_branch2c",
		"res5c_branch2a_relu",
		"res5c_branch2b_relu",
		"scale5c_branch2c"
	]
	for filename in files:
		fn = '../../resnet_results/resnet50_lininterp_{}.out'.format(filename)
		with open(fn) as f:
			for line in f.readlines():
				if 'Batch' not in line and 'produces' not in line and 'accuracy/top' not in line and 'accuracy@' in line:
					if 'accuracy@1' in line:
						top1 = float(line.split()[-1])*100 
					elif 'accuracy@5' in line:
						top5 = float(line.split()[-1])*100
			print filename, top1, top5

def parse_round(filename_suffix):
	print 'Iteration, top1, top5'
	fn = '../../resnet_results/resnet50_lininterp_{}.out'.format(filename_suffix)
	flag = 0
	with open(fn) as f:
		for line in f.readlines():
			if 'Testing net (#0)' in line:
				num_iter = int(line.split()[-4].strip(','))
			if 'Test net output' in line and 'accuracy@' in line:
				if 'accuracy@1' in line:
					top1 = float(line.split()[-1])*100 
				elif 'accuracy@5' in line:
					top5 = float(line.split()[-1])*100
					flag=1
			if flag==1:
				print 'Iteration:{0}'.format(num_iter), top1, top5
				flag = 0
			old_line = line
def parse_time(filename_suffix):
	printDetail = 0
	layers = []
	interp_layers = []
	forwards = []
	forwards_interp = []
	backwards = []
	if printDetail:
		print 'Layer, forward, backward'
	fn = '../../resnet_results/time_resnet50_{}.out'.format(filename_suffix)
	exp_name = fn.split('/')[-1]
	exp_name = exp_name[14:-4]
	flag = 0
	with open(fn) as f:
		for line in f.readlines():
			if 'Iteration' not in line and ('forward' in line or 'backward' in line) and 'ms' in line:
				if 'forward' in line:
					forward = float(line.split()[-2])
					forwards.append(forward)
					if 'interp' in line:
						forwards_interp.append(forward)
						interp_layers.append(str(line.split()[-4]))
				elif 'backward' in line:
					backward = float(line.split()[-2])
					backwards.append(backward)
					layer = str(line.split()[-4])
					layers.append(layer)
					flag=1
			if flag==1:
				if printDetail:
					print layer, forward, backward
				flag = 0
	
	#print interp_layers, forwards_interp
	#print 'Forward time:{0}  Backward time:{1}'.format(sum(forwards[1:-1]),sum(backwards[1:-1]))
	#print 'Interp layers time percentage: {}'.format(sum(forwards_interp)/sum(forwards[1:-1]))
	print '{0} {1} {2} {3}'.format(exp_name,sum(forwards[1:-1]),sum(backwards[1:-1]),sum(forwards_interp)/sum(forwards[1:-1]))

############========================= main ================
#parse_perlayer()
#for iterNum in range(31,45):
#	fileSuffix = 'round{}_CUDAv5_newLRpolicyFrom1e-4_35batchsize'.format(iterNum)
#	print fileSuffix
#	parse_round(fileSuffix)
#parse_time('original_train1test1')
#parse_time('original_train1test30')
#parse_time('original_train1test35')
#parse_time('original_train1test40')
#parse_time('original_train30test5')
#parse_time('original_train30test10')
#parse_time('original_train30test15')
#parse_time('original_train30test20')
#parse_time('original_train30test25')
#parse_time('original_train30test30')
#parse_time('original_train30test35')
#parse_time('original_train30test50')
#parse_time('original_train33test1')
#iterRange = [1] + [2] + range(31,45)
#for iterNum in iterRange:
#	parse_time('lininterp_round{}_interpCUDAv5_train30test50'.format(iterNum))
#for iterNum in iterRange:
#	parse_time('lininterp_round{}_interpCUDAv5_train33test1'.format(iterNum))

parse_time('original_cpu')
iterRange = [1] + [2] + range(31,45)
for iterNum in iterRange:
	parse_time('lininterp_round{}_interpCUDAv5_train33test1_cpu'.format(iterNum))

