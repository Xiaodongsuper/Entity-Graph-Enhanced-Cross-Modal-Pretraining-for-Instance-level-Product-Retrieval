import json
import re
import tqdm

def read_json(file):
    f=open(file,"r",encoding="utf-8").read()
    return json.loads(f)

data = read_json('./dataset_tongkuan/tongkuan_train_title.json')

product_1m = {}
f = open('./dataset_tongkuan/null_entity.txt','a')
f1 = open('./dataset_tongkuan/full_entity.txt','a')
all_entity = []
tag_all = []
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
        if(rs['tag']!='普通词' and rs['tag']!='符号' and rs['tag']!='型号' and rs['tag']!='尺寸规格' and rs['tag']!='数字' and rs['tag']!='单位'  and rs['tag']!='营销服务' and rs['tag']!='材质' and rs['tag']!='地点地域' and rs['tag']!='影视名称' and rs['tag']!='人名' and rs['tag']!='人群' and rs['tag']!='品类' and rs['tag']!='修饰'  and rs['tag']!='否定'  and rs['tag']!='后缀'   and rs['tag']!='风格'   and rs['tag']!='时间季节' and rs['tag']!='代理' and rs['tag']!='新词'  and rs['tag']!='赠送' ): #品质成色' '品牌' '品类' '系列' '修饰' '新词'
            tmp.append(rs['word'])
            tag_all.append(rs['tag'])
            all_entity.append(rs['word'])
    product_1m[data_i]['entity'] = tmp

uni_tag = list(set(tag_all))
with open('./dataset_tongkuan/uni_tag.txt','a') as uni_t:
    for line in uni_tag:
     uni_t.write(line+ '\n')
uni_tag=set(tag_all)
dict={}
for item in uni_tag:
    dict.update({item:tag_all.count(item)})
print(dict)

out_uni = open('./dataset_tongkuan/uni_train_entity.txt','a')
lines_seen = set()
for line in all_entity:
    if line not in lines_seen:
        out_uni.write(line+ '\n')
        lines_seen.add(line)
with open("./dataset_tongkuan/train_entity_data.json","w") as f:
    json.dump(product_1m,f)