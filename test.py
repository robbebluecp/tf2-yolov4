
import models


model = models.YOLO()()
# a = (250, len(model.layers) - 3)[2 - 1]
print(len(model.layers))
