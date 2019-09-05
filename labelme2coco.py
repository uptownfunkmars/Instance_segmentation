import cv2 as cv
import json
import os
import numpy as np

co = {}
co['info'] = {}
co['images'] = []
co['license'] = []
co['annotations'] = []
co['categories'] =[]

dir = r"F:\ZZC\check_gn"
sav_dir = r"F:\ZZC"
FileList = os.listdir(dir)


# co info
co['info']['year'] = 2019
co['info']['version'] = "Shining_3d_1.0"
co['info']['description'] = "Tooth_instance_Segmentation"
co['info']['contributor'] = "Shining_3d"
co['info']['url'] = ""
co['info']['date_created'] = "2019/09/04"

jsonList = []
for i in FileList:
    if i.split('.')[1] == 'json':
        jsonList.append(i)


for i in range(len(jsonList)):
    image = {}
    license = {}

    filename = jsonList[i].split('.')[0] + '.jpg'
    img = cv.imread(dir + '\\' + filename)
    height, width = img.shape[0], img.shape[1]

    ID = i + 1

    image['id'] = ID
    image['width'] = width
    image['height'] = height
    image['file_name'] = filename
    image['license'] = ""
    image['flickr_url'] = ""
    image['coco_url'] = ""
    image['date_captured'] = ""

    license['id'] = ID
    license['name'] = filename
    license['url'] = ""

    co['images'].append(image)
    co['license'].append(license)


classes = []
for i in range(len(jsonList)):
    with open(dir + '\\' + jsonList[i], 'r') as fp:
        D = json.load(fp)
    for j in range(len(D['shapes'])):
        if D['shapes'][j]['label'] not in classes:
            classes.append(D['shapes'][j]['label'])


for i, c in enumerate(classes):
    category = {}
    category['id'] = i + 1
    category['name'] = c
    category['supercategory'] = c

    co['categories'].append(category)

classes_d = {}
for i, C in enumerate(classes):
    classes_d[C] = i + 1

cur = 1
for i in range(len(jsonList)):
    with open(dir + '\\' + jsonList[i], 'r') as fp:
        D = json.load(fp)

    for j in range(len(D['shapes'])):
        temp = {}

        points = np.array(D['shapes'][j]['points'])
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)

        temp['id'] = cur
        cur = cur + 1

        temp['image_id'] = i
        temp['category_id'] = classes_d[D['shapes'][j]['label']] if D['shapes'][j]['label'] in classes else 0
        temp['segmentation'] = [list(np.asarray(D['shapes'][j]['points'])[:, 0].astype(np.float)),
                                list(np.asarray(D['shapes'][j]['points'])[:, 1].astype(np.float))]
        temp['area'] = 1
        temp['bbox'] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        temp['iscrowd'] = 0

        co['annotations'].append(temp)

print(co)
with open(sav_dir + '\\' + 'instance_train_2019_sh3d.json', 'w', encoding='utf-8') as fp:
    json.dump(co, fp, ensure_ascii=False)
#
#
#
#
#
# # filename = jsonList[0].split('.')[0] + '.jpg'
