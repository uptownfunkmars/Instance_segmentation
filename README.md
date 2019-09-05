# Instance_segmentation


可视化校验由labelme分割得到的json文件，同时自动生成bounding_box坐标并添加到labelme生成的json文件中。

# jb.py

批量将json中label为“_”转化为“-”。
例如：
  “other_tooth” -> "other-tooth"

# labelme2coco.py

将labelme标记生成的json文件，转化为可被COCO API识别的json文件。
需要注意的是：
在商汤开源的 mmdetection 中需要在 'annotation' 中包含 'area' 的信息，并且如果 'area' 对应的值小于等于0，会导致生成 dataloader 中出错。
其次，在 COCO 官方 instances_train217.json 文件中 senmentation 部分中包含的信息，并不是将 polygon 的坐标分为x与两列进行保存，而是保存为一个 list.

# mmdetection中使用自己的数据

需要将mmdetection/mmdet/dataset/coco.py中的classes更改为自己的类别标签。

需要将mmdetection/mmdet/core/evaluation/class_names.py修改coco_classes数据集类别.

需要将configs/faster_rcnn_r50_fpn_1x.py中的model字典中的num_classes、data字典中的img_scale和optimizer中的lr(学习率)，修改为自己的类别数(需要加一，将BG包含进去），image scale 改为自己的图像尺寸。

# 例如：
  将labelme中保存的 polygan 转化为能被 COCO api 识别的形式如下
            [[x1, y1], [x2, y2]] ==> [[x1, y1, x2, y2]]
           
