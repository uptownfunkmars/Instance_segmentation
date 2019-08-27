import json
import os

imgspath = r"F:\ZZC\Instance-check\image\check_guona"

#返回路径下的所有文件名， 返回[]
jsons = os.listdir(imgspath)

print(type(jsons))
print(jsons)
print(len(jsons))

count = 0
imgs_jsons = []
for fn in jsons:
    if fn.split('.')[1] == "json":
        count = count + 1
        imgs_jsons.append(fn)

'''
count_n = 0
imgs_jsons_n = []
for i in range(len(jsons)):
    if jsons[i][-4:] == "json":
        count_n = count_n + 1
        imgs_jsons_n.append(jsons[i])

print(imgs_jsons_n[-1])
print(count_n)
'''
print(len(imgs_jsons))


shapes = []
for fname in imgs_jsons:
    if fname != 'd.json':
        DICT = json.load(open(imgspath + '\\' + fname))
        print(type(DICT))
        L = len(DICT['shapes'])

        per_fname_labels = []
        for i in range(L):
            #DICT['shapes'][i]['label'] = DICT['shapes'][i]['label'].spilt('t')[0][-1]
            per_fname_labels.append(DICT['shapes'][i]['label'])
            #print(DICT['shapes'][i]['label'])

        for i in range(L):
            DICT['shapes'][i]['label'] = per_fname_labels[i].split("_")[0] + '-' + per_fname_labels[i].split("_")[1]
        print(DICT)
        with open(imgspath + '\\' + fname, "w") as dump_f:
            DICT_TEMP = json.dump(DICT, dump_f)
        print(type(DICT_TEMP))



