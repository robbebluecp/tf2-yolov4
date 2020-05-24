#INTRODUCTION
This edition of yolov4 frame has been smoothly transferred from tf1+ and keras version. With tf2+ used extensively,
more and more functions has been optimized better. And,  keras has been embedded into tensorflow at the same time and not
supported by official except fixing bugs. Also for more readable, I've made a lot of changes while transferring, such as 
layer aggregation, image augment and some other changes from tf1 to tf2.   
  
【Now, IT IS ONLY SUPPORT TF2+】 


#Installation
git clone https://github.com/robbebluecp/tf2-yolov4.git

#Files
 * data :-> to put some temporary files like jpgs or pngs and so on  
 * font_data :-> a font file for text on pic (will be removed)
 * model_data :-> core config files mainly including class file and anchors file
 * model_train :-> for saving models or weights files
 * tools :-> helper functions 
 * config  :-> configuration file
 * convert :-> for converting weights file to h5 which trained by darknet using tf2+ (do not support tf1+)
 * eval :-> a part of predicting
 * generator :-> a generator of data by loading image files by batch
 * loss :-> core loss function
 * models :-> core yolo4 model
 * predict :-> for predicting
 * prepare :-> prepare config
 * train :-> 😅
 
 #HOW TO PREDICT
 ## If you use yolov4.weights file
 * download yolov4.weights file from [yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)
 * put it into folder model_train/
 * REMEMBER to change 【classes_path = 'model_data/voc_classes.txt'】into 【classes_path = 'model_data/coco_classes.txt'】 in config.py
 * run: python3 convert.py -w model_train/yolov4.weights -m model_train/yolov4.h5
 * then you'll see yolov4.h5 in your model_train folder
 * run: python3 predict -m model_train/yolov4.h5 -i data/test.png
 * then you'll see the following information  
 56 chair 0.98 x:2 y:1037 x:448 y:1628
 0 person 0.83 x:1 y:553 x:335 y:1176
 0 person 0.95 x:257 y:463 x:440 y:943
 0 person 1.00 x:345 y:454 x:788 y:1195  
 and if you have visual interface you can see a pic with rectangles and class name on it
 
 ## If you use your own train model
 * just run: python3 predict.py -m xxx.weights(or xxx.h5) -i xxx.jpg 
 
 #HOW TO TRAIN
 ## Here is an example to show you how to train on voc2007. Images and labels are reorganized so that you can easily divide into your training
 * download voc2007 data from [voc2007.zip](https://github.com/robbebluecp/tf2-yolov4/releases/download/1.0.0/voc2007.zip)
 * unzip it into /opt (of course some other paths are also ok..., if you choose other paths, you have to change 【label_path = '/opt/voc2007/labels.txt'】 into 【label_path = your_path/label.txt】 in config.py)
 * make sure  【classes_path = 'model_data/voc_classes.txt'】 in config.py
 * then ,run python3 train.py
 
 #RESULT
 
 
 
 #RELATIONS
 [http://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects](http://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects])  
 [http://github.com/Ma-Dan/keras-yolo4](http://github.com/Ma-Dan/keras-yolo4)  
 [http://github.com/qqwweee/keras-yolo3](http://github.com/qqwweee/keras-yolo3)  
 [https://arxiv.org/pdf/2004.10934.pdf](https://arxiv.org/pdf/2004.10934.pdf)