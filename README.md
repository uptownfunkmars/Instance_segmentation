# Instance_segmentation

可视化校验由labelme分割得到的json文件，同时自动生成bounding_box坐标并添加到labelme生成的json文件中。

# jb.py

批量将json中label为“_”转化为“-”。
例如：
  “other_tooth” -> "other-tooth"

