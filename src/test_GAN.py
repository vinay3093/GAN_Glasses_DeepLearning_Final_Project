import numpy as np
import time,os,sys
import util
import tensorflow as tf
import data
import graph,warp
import options
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

opt = options.set(training=False)
opt.loadGP = "0/test0_warp5_it50000"  #folder where the resulting images are stored
opt.warpN = 5
opt.loadImage = "dataset/example_test.png" #loading the sample celebrity image

print("building the graph...")
tf.compat.v1.reset_default_graph()

# build graph
with tf.device(opt.GPUdevice):

	# ------ define input data ------
	imageBG = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.H,opt.W,3])
	imageFG = tf.placeholder(tf.float32,shape=[opt.batchSize,opt.H,opt.W,4])
	PH = [imageBG,imageFG]
	pPertFG = opt.pertFG*tf.random_normal([opt.batchSize,opt.warpDim])
    
	# ------ define GP and D ------
	geometric = graph.geometric_multires
    
	# ------ geometric predictor ------
	imageFGwarpAll,_,_ = geometric(opt,imageBG,imageFG,pPertFG)
    
	# ------ composite image ------
	imageCompAll = []
	for l in range(opt.warpN+1):
		imageFGwarp = imageFGwarpAll[l]
		imageComp = graph.composite(opt,imageBG,imageFGwarp)
		imageCompAll.append(imageComp)
        
	# ------ optimizer ------
	varsGP = [v for v in tf.global_variables() if "geometric" in v.name]

# load data
print("loading the test  data...")
path = "dataset"
#glasses = np.load("{0}/glasses.npy".format(path))
glasses = np.load("dataset/glasses.npy".format(path))

# prepare model saver/summary writer
saver_GP = tf.train.Saver(var_list=varsGP)

print("======= Starting Evaluation =======")
#timeStart = time.time()

# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
with tf.Session(config=tfConfig) as sess:
	sess.run(tf.global_variables_initializer())
	util.restoreModel(opt,sess,saver_GP,opt.loadGP,"GP")
	print("start evaluation...")

	# create directories for test image output
	os.makedirs("eval_{0}".format(opt.loadGP),exist_ok=True)
	testImage = util.imread(opt.loadImage)
	batch = data.makeBatchEval(opt,testImage,glasses,PH)
	runList = [imageCompAll[0],imageCompAll[-1]]
	ic0,icf = sess.run(runList,feed_dict=batch)
	for b in range(opt.batchSize):
		util.imsave("eval_{0}/image_g{1}_input.png".format(opt.loadGP,b),ic0[b])
		util.imsave("eval_{0}/image_g{1}_output.png".format(opt.loadGP,b),icf[b])

print("====== Evaluation Completed =======")
