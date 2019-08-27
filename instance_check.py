import numpy as np
import cv2
import os
import scipy.misc
import sys
import matplotlib.pyplot as plt
import json
import imgaug as ia
from labelme import utils


def sigmoid(x):
    s = 1 / (1 + np.exp(x))
    return s


def visulize_result(image, pred, logits=False):
    if logits:
        pred = sigmoid(pred)

    mask = (np.round(pred)).astype(np.uint8)
    mask = mask.squeeze()
    segmap = ia.SegmentationMapOnImage(mask, shape=mask.shape, nb_classes=3)
    colors = [(0, 0, 255), (255, 255, 255), (255, 0, 255)]
    img_add_mask = segmap.draw_on_image(image, colors=colors, alpha=0.3)

    cells = [image, img_add_mask]
    grid_image = ia.draw_grid(cells, cols=2)

    return img_add_mask, grid_image


# label_name_to_value = {'background': 0, "tooth": 1, "gum": 2, "cheek": 3, "lips": 4, "tongue": 5}

# 自行改动的
label_name_to_value = {'background': 0, "healthy_teeth": 1, "preparing_teeth": 2}

'''
#原始
imgspath = r"image"
imgspatn_erro = r"image_error"
imgspatn_correct = r"image_correct"
imgspath_indistinct = r"image_indistinct"
imgs = os.listdir(imgspath)
'''

# 改动
imgspath = r"F:\ZZC\DataSetTest\image"
imgspatn_erro = r"F:\ZZC\DataSetTest\image_error"
imgspatn_correct = r"F:\ZZC\DataSetTest\image_correct"
imgspath_indistinct = r"F:\ZZC\DataSetTest\image_indistinct"
imgs = os.listdir(imgspath)

imgs_jpg = []
imgs_json = []
# 1 找到所有的jpg and json 文件，delete 扩展名并save to imgs_jpg ，save imgs_json
# 2 set到imgs_jpg and imgs_json 的并集 （只对有原图片和json的文件进行操作）
for img in imgs:
    if (img.split('.')[1] == "jpg"):
        imgs_jpg.append(img.split('.')[0])
    else:
        imgs_json.append(img.split('.')[0])
imgs_jpg = list(set(imgs_jpg) & set(imgs_json))

imgs_jpg_json = []
for im in imgs_jpg:
    imgs_jpg_json.append(im + ".jpg")

tem = []

'''
for i, img in enumerate(imgs_jpg_json):
    print(i)
    print(img)
    tem.append(img)
    # TODO  #THIS IS  ？？？
    tem_out = tem[0]

    images1 = plt.imread(os.path.join(imgspath, img))
    h = images1.shape[0]  # shape()返回的是一个3维数组（height，weight，c）c是颜色通道指grb=3
    w = images1.shape[1]
    print("---", h, w, images1.shape[2])

    # b,g,r = cv2.split(images1)
    # images2= cv2.merge([r,g,b])
    images_mask = json.load(open(os.path.join(imgspath, img.split('.')[0] + '.json')))

    images_mask = utils.shapes_to_label((h, w), images_mask['shapes'], label_name_to_value)   #绘制掩码

    #print(images_mask > 3 - 1)
    #print(images_mask)

    images_mask[images_mask > 3 - 1] = 0

    images_mask1 = visulize_result(images1, images_mask)[0]        #可视化
    hmerge = np.hstack((images1, images_mask1))                    #进行堆叠

    b, g, r = cv2.split(hmerge)                                    #重新合并通道
    hmerge = cv2.merge([r, g, b])

    cv2.namedWindow(img, 0)                                        #进行显示
    cv2.resizeWindow(img, 900, 700)
    cv2.imshow(img, hmerge)
    cv2.moveWindow(img, 150, 150)

    k = cv2.waitKey(0)
    print(k)
    path_json = os.path.join(imgspath, img.split('.')[0] + '.json')
    path1 = os.path.join(imgspatn_correct, img.split('.')[0] + '.json')
    path2 = os.path.join(imgspatn_erro, img.split('.')[0] + '.json')
    path3 = os.path.join(imgspath_indistinct, img.split('.')[0]+'.json')
    with open(path_json, 'r') as f:
        temp = json.loads(f.read())

    # # k = cv2.waitKey(0) & 0xFF  # 64位机器
    if k == ord('7'):         # esc: True ,display next
        scipy.misc.toimage(images1, cmin=0.0, cmax=1.0).save(os.path.join(imgspatn_correct, img.split('.')[0] + '.jpg'))
        with open(path1, 'w', encoding='utf-8') as json_file:
            json.dump(temp, json_file, ensure_ascii=False)
        if i > 0:
            os.remove(os.path.join(imgspath, tem_out))
            os.remove(os.path.join(imgspath, tem_out.split('.')[0] + '.json'))
        cv2.destroyAllWindows()

    elif k == ord('8'):   # space: error ,save to image_erro
        scipy.misc.toimage(images1, cmin=0.0, cmax=1.0).save(os.path.join(imgspatn_erro, img.split('.')[0] + '.jpg'))
        with open(path2, 'w', encoding='utf-8') as json_file:
            json.dump(temp, json_file, ensure_ascii=False)
        if i > 0:
            os.remove(os.path.join(imgspath, tem_out))
            os.remove(os.path.join(imgspath, tem_out.split('.')[0] + '.json'))
        cv2.destroyAllWindows()

    elif k == ord('q'):   # q: indistinct,save to  image_indistinct
        scipy.misc.toimage(images1, cmin=0.0, cmax=1.0).save(os.path.join(imgspath_indistinct, img.split('.')[0] + '.jpg'))
        with open(path3, 'w', encoding='utf-8') as json_file:
            json.dump(temp, json_file, ensure_ascii=False)
        if i > 0:
            os.remove(os.path.join(imgspath, tem_out))
            os.remove(os.path.join(imgspath, tem_out.split('.')[0] + '.json'))
        cv2.destroyAllWindows()

    else:  # other :exit
        print(k)
        cv2.destroyAllWindows()
        sys.exit()

    if i > 0:
        del tem[0]
'''

###############################################################################################

for i, img in enumerate(imgs_jpg_json):
    print(i)
    print(img)
    tem.append(img)
    # TODO  #THIS IS  ？？？
    tem_out = tem[0]

    images1 = plt.imread(os.path.join(imgspath, img))
    h = images1.shape[0]  # shape()返回的是一个3维数组（height，weight，c）c是颜色通道指grb=3
    w = images1.shape[1]
    print("---", h, w, images1.shape[2])

    # b,g,r = cv2.split(images1)
    # images2= cv2.merge([r,g,b])
    images_mask = json.load(open(os.path.join(imgspath, img.split('.')[0] + '.json')))

    # 增加一个读入的dict的引用方便后续取b-box操作
    images_mask_1 = images_mask

    images_mask = utils.shapes_to_label((h, w), images_mask['shapes'], label_name_to_value)  # 绘制掩码
    # print(images_mask > 3 - 1)
    # print(images_mask)
    images_mask[images_mask > 3 - 1] = 0
    images_mask1 = visulize_result(images1, images_mask)[0]  # 可视化，此时已经能够使用imshow输出

    # images_mask1 = cv2.rectangle(images_mask1, (30, 30), (70, 70), (0, 0, 255), thickness=2)  #绘制b-box

    # b-box存储每个对象的b-box坐标
    b_box = []
    cv2.namedWindow("temp", 0)
    LEN = len(images_mask_1['shapes'])

    # vector是image_mask_1的copy
    vector = images_mask_1['shapes'].copy()
    # 对图上的每个对象的b-box进行校验
    for j in range(LEN):
        points = np.array(vector[j]['points'])

        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)

        print("\n")
        print("原始坐标")
        print("(", int(x1), int(y1), ")")
        print("(", int(x2), int(y2), ")")

        # 第一次显示
        # 复制两个可显示的掩码
        images_mask_temp = images_mask1.copy()
        images_mask_temp_1 = images_mask1.copy()

        # 绘制b-box
        images_mask_temp_rec = cv2.rectangle(images_mask_temp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255),
                                             thickness=2)
        cv2.imshow("temp", images_mask_temp)

        print("b-box是否需要更改？(yes:y, no: n)")
        k1 = cv2.waitKey(0)
        print("Your input is %c" % k1)
        print("##########################\n"
              "#     w 向上移动          #\n"
              "#     s 向下移动          #\n"
              "#     a 向左移动          #\n"
              "#     d 向左移动          #\n"
              "#     space 切换坐标点    #\n"
              "#     y 退出             #\n"
              "##########################\n")

        if k1 == ord("y"):
            x = int(x1)
            y = int(y1)
            tag = True  # 代表(x, y)是否代表(x1, y1)

            cur_pos = 1
            count_space = 0
            while (1):
                print("当前坐标为(x%d, y%d)" % (cur_pos, cur_pos))
                print("开始更改坐标...................")

                k2 = cv2.waitKey(0)

                if k2 == ord('w'):
                    print("(x%d, y%d)向上移动" % (cur_pos, cur_pos))
                    y = y - 50
                elif k2 == ord('s'):
                    print("(x%d, y%d)向下移动" % (cur_pos, cur_pos))
                    y = y + 50
                elif k2 == ord('a'):
                    print("(x%d, y%d)向左移动" % (cur_pos, cur_pos))
                    x = x - 50
                elif k2 == ord('d'):
                    print("(x%d, y%d)向右移动" % (cur_pos, cur_pos))
                    x = x + 50
                elif k2 == ord(' '):
                    count_space = count_space + 1

                    if (count_space % 2) != 0:
                        cur_pos = 2
                        tag = False
                        x1 = x
                        y1 = y
                        x = int(x2)
                        y = int(y2)
                    else:
                        cur_pos = 1
                        tag = True
                        x2 = x
                        y2 = y
                        x = int(x1)
                        y = int(y1)
                    print("坐标切换为(x%d, y%d)" % (cur_pos, cur_pos))
                elif k2 == ord('y'):
                    print("退出")
                    break

                if tag:
                    x1 = x
                    y1 = y
                else:
                    x2 = x
                    y2 = y

                print("更改后为：")
                print("(", int(x1), int(y1), ")")
                print("(", int(x2), int(y2), ")")

                images_mask_temp = images_mask_temp_1.copy()
                images_mask_temp_rec = cv2.rectangle(images_mask_temp, (int(x1), int(y1)), (int(x2), int(y2)),
                                                     (0, 0, 255), thickness=2)  # 绘制b-box
                cv2.imshow("temp", images_mask_temp_rec)
        cv2.destroyWindow("temp")

        print("\n")
        print("更改后的坐标")
        print(int(x1), int(y1))
        print(int(x2), int(y2))
        print("\n")
        b_box.append([[x1, y1, x2, y2]])

    hmerge = np.hstack((images1, images_mask1))  # 进行堆叠
    # hmerge = images_mask1  # 进行堆叠

    b, g, r = cv2.split(hmerge)  # 重新合并通道
    hmerge = cv2.merge([r, g, b])

    cv2.namedWindow(img, 0)  # 进行显示
    cv2.resizeWindow(img, 900, 700)
    cv2.imshow(img, hmerge)
    # cv2.moveWindow(img, 150, 150)

    k = cv2.waitKey(0)
    print(k)
    path_json = os.path.join(imgspath, img.split('.')[0] + '.json')
    path1 = os.path.join(imgspatn_correct, img.split('.')[0] + '.json')
    path2 = os.path.join(imgspatn_erro, img.split('.')[0] + '.json')
    path3 = os.path.join(imgspath_indistinct, img.split('.')[0] + '.json')

    # 读入json
    with open(path_json, 'r') as f:
        temp = json.loads(f.read())

    # 此处完成b-box的插入
    temp['b_box'] = b_box
    print(temp)

    # # k = cv2.waitKey(0) & 0xFF  # 64位机器
    if k == ord('7'):  # esc: True ,display next
        scipy.misc.toimage(images1, cmin=0.0, cmax=1.0).save(os.path.join(imgspatn_correct, img.split('.')[0] + '.jpg'))
        with open(path1, 'w', encoding='utf-8') as json_file:
            json.dump(temp, json_file, ensure_ascii=False)  # 此处需要完成json b-box坐标更新操作
        if i > 0:
            os.remove(os.path.join(imgspath, tem_out))
            os.remove(os.path.join(imgspath, tem_out.split('.')[0] + '.json'))
        cv2.destroyAllWindows()

    elif k == ord('8'):  # space: error ,save to image_erro
        scipy.misc.toimage(images1, cmin=0.0, cmax=1.0).save(os.path.join(imgspatn_erro, img.split('.')[0] + '.jpg'))
        with open(path2, 'w', encoding='utf-8') as json_file:
            json.dump(temp, json_file, ensure_ascii=False)  # 此处需要完成json b-box坐标更新操作
        if i > 0:
            os.remove(os.path.join(imgspath, tem_out))
            os.remove(os.path.join(imgspath, tem_out.split('.')[0] + '.json'))
        cv2.destroyAllWindows()

    elif k == ord('q'):  # q: indistinct,save to  image_indistinct
        scipy.misc.toimage(images1, cmin=0.0, cmax=1.0).save(
            os.path.join(imgspath_indistinct, img.split('.')[0] + '.jpg'))
        with open(path3, 'w', encoding='utf-8') as json_file:
            json.dump(temp, json_file, ensure_ascii=False)  # 此处需要完成json b-box坐标更新操作
        if i > 0:
            os.remove(os.path.join(imgspath, tem_out))
            os.remove(os.path.join(imgspath, tem_out.split('.')[0] + '.json'))
        cv2.destroyAllWindows()

    else:  # other :exit
        print(k)
        cv2.destroyAllWindows()
        sys.exit()

    if i > 0:
        del tem[0]

# scipy.misc.to
