import shutil
import os

matched_rs = open('./feature/results/vil_feature_retrieval_id_list.txt','r')

rs_num = 0
for rs_i in matched_rs.readlines():
    retrieval_rs = rs_i.split('\n')[0].split(',')
    if(os.path.exists('./several_mathed_rs/'+retrieval_rs[0])):
        pass
    else:
        os.makedirs('./several_mathed_rs/'+retrieval_rs[0])
    shutil.copy('/data2/xiaodong/Product1m/all_data/images/'+retrieval_rs[0]+'.jpg', './several_mathed_rs/'+retrieval_rs[0]+'/!'+retrieval_rs[0]+'.jpg')
    tmp = 1
    for img_i in retrieval_rs[1:]:
        tmp_img_num = 0
        shutil.copy('/data2/xiaodong/Product1m/all_data/images/'+img_i+'.jpg', './several_mathed_rs/'+retrieval_rs[0]+'/'+str(tmp)+'.jpg')
        tmp +=1
        if(tmp>20):
            break
    rs_num +=1 
    if(rs_num>2000):
        break