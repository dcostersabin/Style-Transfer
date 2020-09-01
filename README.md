# Style-Transfer

## Tensorflow implementation of Image Style Transfer Using Convolutional Neural Network
  ##### Based on (Research Paper) : https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
 
 This is implementation of neural style transfer in tensorflow. It is built on top of  __VGG-19__ excluding the top layers for predictiong the image outcome
 The output is only trained for 100 iterations which converses beautifully. The loss parmeters and the layers to extract the features can be tweaked for your 	 preference. Beauty is subjective so whatever you think is best.
 
 ### Requirements:
  1. Tensorflow 
  2. Numpy 
 
 ##### How To Run ?
 1. Install above mentioned dependencies 
 2. Save content and style image as content.jpeg and style.jped on project root directory
 3. Initialize the hyperparameters as you prefer
 4. Go to terminal and write ` python3 StyleTransfer.py` press enter 
 5. Let it train for the iterations defined
 6. Check the generated image in output dir
 7. Enjoy Learning 
 
 
### Content Image:
  Image Feeded To Network :
![alt text](https://github.com/dcostersabin/Style-Transfer/blob/master/content.jpeg)

### Style Image:
  Style Image Feeded To Network :
  ![alt text](https://github.com/dcostersabin/Style-Transfer/blob/master/style.jpeg)
  
   ### Neural Networks Output:
  
  	First 10 itration:
  ![alt text](https://github.com/dcostersabin/Style-Transfer/blob/master/outputs/10.jpg)
  
  
  	Image predicted by the neural netowrk after 100 iterations:
  ![alt text](https://github.com/dcostersabin/Style-Transfer/blob/master/outputs/100.jpg)
