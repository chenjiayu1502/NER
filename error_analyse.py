#encoding=utf-8
import json

# file='result/0612_long_pred_lstmlstm.txt'
file='./result/0612_long_pred_lstmlstm.txt'
# file='./result/0617_long_pred_lstmcrf.txt'
def label_wrong(file):
	f=open(file,'r')
	lines=f.readlines()
	f.close()
	label_dic={}
	for line in lines:
		line=line.rstrip().split(' ')
		if len(line)==3 and line[1]!=line[2]:
			if line[2] not in label_dic:
				label_dic[line[2]]={line[1]:1}
			else:
				if line[1] not in label_dic[line[2]]:
					label_dic[line[2]][line[1]]=1
				else:
					label_dic[line[2]][line[1]]+=1
	for k,v in label_dic.items():
		print('key==',k)
		for k1,v1 in v.items():
			print('\t'+k1+'\t'+str(v1))
#针对每一个属性，统计标签类别错误情况，Ｏ也需要统计
# {
# 	SX1:{
# 		labelb:{
# 			label1:cnt
# 			label2:cnt
# 			...
# 		},num
# 		labeli{
# 			label1:cnt
# 			label2:cnt
# 			...
# 		},num

# 	},num
# 	SX2:{
# 		...

# 	},num
# 	O:{
# 		label1:cnt
# 		label2:cnt
# 	}
# }
def entity_wrong(file):
	
	f=open(file,'r')
	lines=f.readlines()
	f.close()
	label_o=[{},0]
	label_dic={}
	for line in lines:
		line=line.rstrip().split(' ')
		if len(line)==3 and line[1]!=line[2]:
			#o for label_o
			if line[2]=='O':
				if line[1] not in label_o[0]:
					label_o[0][line[1]]=1
				else:
					label_o[0][line[1]]+=1
				label_o[1]+=1
				continue

			#others
			sx=line[2].split('-')[1]
			if sx not in label_dic:
				label_dic[sx]=[{},1]
			else:
				label_dic[sx][1]+=1
			
			if line[2] not in label_dic[sx][0]:
				label_dic[sx][0][line[2]]=[{line[1]:1},1]
			else:
				label_dic[sx][0][line[2]][1]+=1
				if line[1] not in label_dic[sx][0][line[2]][0]:
					label_dic[sx][0][line[2]][0][line[1]]=1
				else:
					label_dic[sx][0][line[2]][0][line[1]]+=1
	# label_dic={'a':1,'b':2}
	cnt=entity_cnt(file)
	# print(cnt)
	for k,v in label_dic.items():
		# print(k,cnt[k])
		v.append(cnt[k])



	crf=['XINGHAO','BACKCAMERA','ROM','STORECARD']
	label_dic=sorted(label_dic.items(),key=lambda x:x[1][1])
	for i in range(len(label_dic)):
		# if label_dic[i][0] not in crf:
		# 	continue
		ensum=0
		for k,v in label_dic[i][1][2].items():
			ensum+=label_dic[i][1][2][k]
		# print(label_dic[i][0]+'\t'+str(label_dic[i][1][1])+'\t'+str(ensum))
		
		for k,v in label_dic[i][1][0].items():

			# print(k)
			print(label_dic[i][0]+'\t'+str(label_dic[i][1][1])+'\t'+str(ensum)+'\t'+k)
			for k1,v1 in v[0].items():
				print('\t'*4+k1+'\t'+str(v1))
	# print(json.dumps(label_dic))
	# with open('data.json', 'w') as json_file:
	# 	json_file.write(json.dumps(label_dic))

def space_count(file):
	f=open(file,'r')
	lines=f.readlines()
	f.close()
	cnt=0
	for line in lines:
		line=line.strip().split(' ')
		for i in range(len(line)):
			if line[i]==' ':
				cnt+=1
	print(cnt)
def entity_cnt(file):
	f=open(file,'r')
	lines=f.readlines()
	f.close()
	cnt={}
	for line in lines:
		line=line.strip().split(' ')
		if len(line)==3 and line[-1]!='O':
			sx=line[-1].split('-')[-1]
			if sx not in cnt:
				cnt[sx]={}
			if line[-1] not in cnt[sx]:
				cnt[sx][line[-1]]=1
			else:
				cnt[sx][line[-1]]+=1
	# print(cnt)
	return cnt

# entity_cnt(file)	


def space():
	s1='　'
	s2=' '
	s3=' '
	print(ord(s1))#12288
	print(ord(s2))#32
	print(ord(s3))#160
	print(s1.isspace())#True
	print(s2.isspace())#True
	print(s3.isspace())#True

	# space_list=[]
	# space_id=[]
	# f=open('./data0520/x_train.txt','r')
	# for line in f.readlines():
	# 	line=list(line.strip())
	# 	for i in range(len(line)):
	# 		if line[i].isspace():
	# 			a=ord(line[i])
	# 			if a not in space_id:
	# 				space_id.append(a)
	# 				space_list.append('#'+line[i]+'#')
	# f.close()
	# print(space_id)
	# print(space_list)
	# for s in space_list:
	# 	print(s.encode('utf-8').decode('utf-8'))

	s1_cnt=0
	# s3_cnt=0
	# f=open('./data0520/x_train.txt','r')
	# for line in f.readlines():
	# 	line=list(line.strip())
	# 	for i in range(len(line)):
	# 		if line[i]==s1:
	# 			s1_cnt+=1
	# 			break
				
	# 		if line[i]==s3:
	# 			s3_cnt+=1
	# 			break

	# f.close()
	# print('s1==',s1_cnt)
	# print('s3==',s3_cnt)
	s='量 键 　 　 虽 然'
	s1=s.split()
	print(len(s1))  # 4
	s2=s.split(' ')
	print(len(s2))  #  7


space()		

# label_wrong(file)
# entity_wrong(file)
# file='./data0520/x_train.txt'
# space_count(file)
#117+1425



