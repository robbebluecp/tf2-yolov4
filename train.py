
"""

This module is disable for training without pre-training weights.
So, just make a chance for learning yolo series net structure
If you wants train your own model with your own data, step to :
[https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects]
or
[https://github.com/robbebluecp/darknet] also the same hhh....

"""


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


import loss
import config
import models
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from generator import data_generator
from tensorflow.keras.callbacks import *


anchors = config.anchors
class_names = config.classes_names
num_anchors = len(anchors)
num_classes = len(class_names)
class_mapping = dict(enumerate(class_names))
class_mapping = {class_mapping[key]: key for key in class_mapping}

model_yolo = models.YOLO()()
model_yolo.summary()

f = open(config.label_path)
label_lines = f.readlines()

train_lines = label_lines[:int(len(label_lines) * config.validation_split)]
valid_lines = label_lines[int(len(label_lines) * config.validation_split):]



h, w = config.image_input_shape
y_true = [Input(shape=(h // config.scale_size[l], w // config.scale_size[l], num_anchors // 3, num_classes + 5)) for l in range(3)]


model_loss = Lambda(function=loss.yolo4_loss,
                    output_shape=(1,),
                    name='yolo_loss',
                    arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5, 'use_diou_loss': True,
                               })([*model_yolo.output, *y_true])

logging = TensorBoard()
checkpoint = ModelCheckpoint(filepath='model_train/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                             monitor='val_loss',
                             save_weights_only=True,
                             save_best_only=True,
                             period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


model = Model([model_yolo.input, *y_true], model_loss)
model.compile(optimizer=Adam(1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
model.fit_generator(generator=data_generator(label_lines=train_lines,
                                             batch_size=config.batch_size,
                                             input_shape=config.image_input_shape,
                                             anchors=anchors,
                                             num_classes=num_classes),
                    validation_data=data_generator(label_lines=valid_lines,
                                             batch_size=config.batch_size,
                                             input_shape=config.image_input_shape,
                                             anchors=anchors,
                                             num_classes=num_classes),
                    steps_per_epoch=len(label_lines) // config.batch_size,
                    validation_steps=int(len(label_lines) * config.validation_split),
                    epochs=config.epochs,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping]
                    )
model_yolo.save('model_train/model_yolo.h5')