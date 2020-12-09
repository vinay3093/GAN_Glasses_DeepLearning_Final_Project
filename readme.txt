#Executables of this project

Prerequisite:
	python3.7+, tensorflow-gpu2.3+.
	
1. Training the Model 

 - Using cmd / Powershell enter into the project folder.
 
 - execute 'python train_pretrain.py' which creates the pretrained models.
 
 - then execute 'python train_GAN.py'

2. Testing the Model

 - execute 'python test_GAN.py'
 
 - then the result is saved in eval_0 folder.

3. warp.py
- this file helps in implementing the warp operation. In other words,it helps in getting the warped Images.

4. graph.py
- this file helps in network architecture. Network consists of two parts
	a)STN generator 
	b)Discriminator

5. util.py
- this file helps in saving the models and the result images.


