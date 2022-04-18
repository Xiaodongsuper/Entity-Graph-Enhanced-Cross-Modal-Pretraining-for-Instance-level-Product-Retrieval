import json
import re
import tqdm

def read_json(file):
    f=open(file,"r",encoding="utf-8").read()
    return json.loads(f)

data = read_json('./dataset/gallery.json')

product_1m = {}
f = open('./dataset/null_entity.txt','a')
f1 = open('./dataset/full_entity.txt','a')
all_entity = []
for data_i in tqdm.tqdm(data):
    tmp = []
    product_1m[data_i] = {}
    rs_data_i = json.loads(data[data_i])
    #product_1m[data[data_i]]['title'] = data_i['title']
    if(rs_data_i['status']!='OK'):
        f.write(data_i+'\n')
        continue
    else:
        f1.write(data_i+'\n')
    for rs in rs_data_i['result']:
        if(rs['tag']!='普通词' and rs['tag']!='符号' and rs['tag']!='型号' and rs['tag']!='尺寸规格' and rs['tag']!='数字' and rs['tag']!='单位'  and rs['tag']!='营销服务' and rs['tag']!='材质' and rs['tag']!='地点地域' and rs['tag']!='影视名称'): #品质成色' '品牌' '品类' '系列' '修饰' '新词'
            tmp.append(rs['word'])
            all_entity.append(rs['word'])
    product_1m[data_i]['entity'] = tmp


out_uni = open('./dataset/uni_entity.txt','a')
lines_seen = set()
for line in all_entity:
    if line not in lines_seen:
        out_uni.write(line+ '\n')
        lines_seen.add(line)
with open("./dataset/gallery_entity_data.json","w") as f:
    json.dump(product_1m,f)