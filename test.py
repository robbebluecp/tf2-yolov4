# import cv2 as cv
# # from tools.utils_image import Augment
# #
# #
# # while 1:
# #     img_path1 = 'data/000030.jpg'
# #     boxes1 = [[36, 205, 180, 289], [51, 160, 150, 292], [295, 138, 450, 290]]
# #
# #     img1 = cv.imread(img_path1)
# #
# #     f = open('model_data/labels.txt')
# #     label_lines = f.readlines()[:]
# #
# #     new_image, new_boxes = Augment(img=img1, boxes=boxes1, img_info_list=label_lines)()
# #
# #     for box in new_boxes:
# #         x1, y1, x2, y2, _ = box
# #         new_image = cv.rectangle(new_image, (x1, y1), (x2, y2), (0, 250, 0))
# #     cv.imshow('', new_image)
# #     cv.waitKey()
# #     cv.destroyAllWindows()
# #
# # exit()


print(len("""从第一版的yolov3（http://github.com/qqwweee/keras-yolo3）在这位q神翻译出来后，在下一直跟进yolo的发展，两年前第一次迁移了q神的keras版。最近keras版的yolov4（http://github.com/Ma-Dan/keras-yolo4）也问世了。由于tf发展到了tf2+，很多模型建立过程、命名规则、文件读取方法以及keras的支持等，都做了非常大的调整，再加上该版本的代码是延续yolov3的代码，没有使用论文的很多tricks，加上历史遗留代码存在很多的不可读因素和局部地方的小bug。因此，基于以上两点考虑，在下联合一位cv从业同学完成了基于tf2版本的、用keras编写的yolov4.

        请收下传送门：https://github.com/robbebluecp/tf2-yolov4

        对于这版的yolov4，我们做了如下优化：

     （1）数据增强。我们在之前的resize、色彩调整、旋转的基础上，增加了mixup、mosaic、任意角度旋转（不建议用任意角           

               度）、背景填充、pixel等数据增强策略；

     （2）模型整合。对yolo整体网络结构和局部结构做详细拆分和更详细的整合。如darknet、spp、pan等；

     （3）loss优化。ciou优化、loss代码优化；

     （4）convert调整。tf1+和tf2+对darknet权重文件的读取，从二进制流和命名方法上都有很大不的不同，tf2+转换非常快，且

               跟tf1不能兼容；

     （5）config配置文件取代动态传参；

     （6）尽可能使np和tf分离，让训练和预测在一定程度上提速；

     （7）生成器兼容、数据增强模块可扩展等其他优化。

        我用voc2007数据集在V100显卡上训练到160+个epochs时，loss和val_loss差不多收敛到13+，预测准确率大约在40%-85%波动。随着预训练模型的加入或者更多epoch的训练，这个值会越来越小。"""))