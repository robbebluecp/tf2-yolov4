


with open('model_train/yolov4.weights', 'rb') as f:
    data = f.read()
    f.close()
    print(data[:40])
    print(data[-40:])

with open('model_train/yolov4_test.weights', 'wb') as f:
    f.write(data[20:] + data[:-20])
    f.close()