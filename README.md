# Instance_segmentation

可视化校验由labelme分割得到的json文件，同时自动生成bounding_box坐标并添加到labelme生成的json文件中。

# jb.py

批量将json中label为“_”转化为“-”。
例如：
  “other_tooth” -> "other-tooth"

# labelme2coco.py

将labelme标记生成的json文件，转化为可被COCO API识别的json文件。
需要注意的是：
在商汤开源的mmdetection中需要在'annotation'中包含'area'的信息，并且如果'area'对应的值小于等于0，会导致生成dataloader中出错。
其次，在COCO官方instances_train217.json文件中senmentation部分中包含的信息，并不是将polygan的坐标分为x与两列进行保存，而是保存为一个list.

# 例如：
  将labelme中保存的 polygan 转化为能被COCO api识别的形式如下
            [[x1, y1], [x2, y2]] ==> [[x1, y1, x2, y2]]
           
