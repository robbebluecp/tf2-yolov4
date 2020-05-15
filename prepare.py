import numpy as np


class PrepareConfig:

    def __init__(self):
        pass

    @staticmethod
    def get_classes(classes_path):
        """loads the classes"""
        with open(classes_path, 'r') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    @staticmethod
    def get_anchors(anchors_path):
        """loads the anchors from a file"""
        with open(anchors_path, 'r') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


