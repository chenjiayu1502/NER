#encoding=utf-8
def get_entity(line):
	former_state='O'
	res=[]
	temp=[]
	for i in range(len(line)):
		now_stage=line[i]
		if now_stage=='O':
			if temp!=[]:
				res.append(temp)
				temp=[]

		elif now_stage=='B':
			if temp!=[]:
				res.append(temp)
				temp=[]
			temp.append(i)

		elif now_stage=='I':
			temp.append(i)
		
		former_state=now_stage
	return res

def get_multi_entity(line):
	former_state='O'
	former_sx=''
	res=[]
	temp=[]
	for i in range(len(line)):
		now_stage=line[i][0]
		now_sx=line[i].split('-')[-1]
		#print(i,temp,former_sx,former_state,now_sx,now_stage)
		if now_stage=='O':
			if temp!=[]:
				temp.append(former_sx)
				res.append(temp)
				temp=[]
				former_sx=''

		elif now_stage=='B':
			if temp!=[]:
				temp.append(former_sx)
				res.append(temp)
				temp=[]
			former_sx=now_sx
			temp.append(i)

		elif now_stage=='I':
			if temp!=[]:
				if former_sx==now_sx:
					temp.append(i)
				else:
					temp.append(former_sx)
					res.append(temp)
					temp=[i]
					former_sx=now_sx
			else:
				former_sx=now_sx
				temp.append(i)
		former_state=now_stage
	return res

def p_r_f1(res1,res2):
	if len(res1)==0 or len(res2)==0:
		return 0.0,0.0,0.0
	else:
		p=0.0
		for i in range(len(res1)):
			if res1[i] in res2:
				p+=1.0
		p=p/len(res1)
		#print('p==',p)
		r=0.0
		for i in range(len(res2)):
			if res2[i] in res1:
				r+=1.0
		r=r/len(res2)
		#print('r==',r)
		if p==0 or r==0:
			return p,r,0.0
		else:
			return p,r,(2*p*r)/(p+r)
def p_r_f1_2(res1,res2):
	if len(res1)==0 or len(res2)==0:
		return 0.0,0.0#,0.0
	else:
		p=0.0
		for i in range(len(res1)):
			if res1[i] in res2:
				p+=1.0
		#p=p/len(res1)
		#print('p==',p)
		r=0.0
		for i in range(len(res2)):
			if res2[i] in res1:
				r+=1.0
		#r=r/len(res2)
		#print('r==',r)
		if p==0 or r==0:
			return p,r#,0.0
		else:
			return p,r#,(2*p*r)/(p+r)

def p_r_f1_for_shuxing(res1,res2):
	dic={}
	if len(res1)==0 or len(res2)==0:
		return dic#0.0,0.0#,0.0
	else:
		for i in range(len(res1)):
			sx=res1[i][-1]
			if sx not in dic:
				dic[sx]=[0,0,0,0]
			dic[sx][1]+=1
			if res1[i] in res2:
				dic[sx][0]+=1
		
		for i in range(len(res2)):
			sx=res2[i][-1]
			if sx not in dic:
				dic[sx]=[0,0,0,0]
			dic[sx][3]+=1
			if res2[i] in res1:
				dic[sx][2]+=1
		return dic



def compute_f1(file1,file2):
	f1=open(file1,'r')
	f2=open(file2,'r')
	lines1=f1.readlines()
	lines2=f2.readlines()
	f1.close()
	f2.close()
	assert len(lines1)==len(lines2)
	p_list=[]
	r_list=[]
	f1_list=[]
	for i in range(len(lines1)):
		line1=lines1[i].strip().split()
		#print(line1)
		line2=lines2[i].strip().split()
		#print(line2)
		if line1==line2:
			f1=1.0
		#res1=get_entity(line1)
		#res2=get_entity(line2)
		res1=get_multi_entity(line1)
		res2=get_multi_entity(line2)
		print(res1)
		print('---------')
		print(res2)
		
		
		p,r,f1=p_r_f1(res1,res2)
		print(p,r,f1)
		p_list.append(p)
		r_list.append(r)
		f1_list.append(f1)
		#break
		
	
	print(sum(p_list)/len(p_list))
	print(sum(r_list)/len(r_list))
	print(sum(f1_list)/len(f1_list))
	
def compute_f1_2(file1,file2):
	f1=open(file1,'r')
	f2=open(file2,'r')
	lines1=f1.readlines()
	lines2=f2.readlines()
	f1.close()
	f2.close()
	assert len(lines1)==len(lines2)
	p_list=[]
	r_list=[]
	len_p_list=[]
	len_r_list=[]
	cnt=0
	for i in range(len(lines1)):
		line1=lines1[i].strip().split()
		#print(line1)
		line2=lines2[i].strip().split()
		if len(line1)<450 or len(line1)>500:
			continue
		cnt+=1
		#print(line2)
		if line1==line2:
			f1=1.0
		#res1=get_entity(line1)
		#res2=get_entity(line2)
		res1=get_multi_entity(line1)
		res2=get_multi_entity(line2)
		# print(res1)
		# print('---------')
		#print(res2)
		len_p_list.append(len(res1))
		len_r_list.append(len(res2))
		
		
		p,r=p_r_f1_2(res1,res2)
		if p>0 and r>0:
			print(p/len(res1),r/len(res2))
		p_list.append(p)
		r_list.append(r)
		
		#break
	
	p_all=sum(p_list)/sum(len_p_list)
	r_all=sum(r_list)/sum(len_r_list)
	# print(p_all,r_all)
	# print(2*p_all*r_all/(p_all+r_all))
	# print(cnt)
	print(p_all,r_all,2*p_all*r_all/(p_all+r_all),cnt)
	
	#print(sum(f1_list)/len(f1_list))

def compute_f1_shuxinglen(file1,file2):
	f1=open(file1,'r')
	f2=open(file2,'r')
	lines1=f1.readlines()
	lines2=f2.readlines()
	f1.close()
	f2.close()
	assert len(lines1)==len(lines2)
	p_list=[]
	r_list=[]
	len_p_list=[]
	len_r_list=[]
	cnt=0
	dic={}
	for i in range(len(lines1)):
		line1=lines1[i].strip().split()
		#print(line1)
		line2=lines2[i].strip().split()
		#print(line2)
		if line1==line2:
			f1=1.0
		#res1=get_entity(line1)
		#res2=get_entity(line2)
		res1=get_multi_entity(line1)
		res2=get_multi_entity(line2)
		if len(res2) not in dic:
			dic[len(res2)]=[[],[],[],[],0]
		#print(res2)
		dic[len(res2)][1].append(len(res1))
		dic[len(res2)][3].append(len(res2))
		dic[len(res2)][-1]+=1
		
		p,r=p_r_f1_2(res1,res2)
		#if p>0 and r>0:
		#	print(p/len(res1),r/len(res2))
		dic[len(res2)][0].append(p)
		dic[len(res2)][2].append(r)
		
		#break
	
	for i in range(20):
		if i in dic:
			p=sum(dic[i][0])/(sum(dic[i][1])+0.00001)
			r=sum(dic[i][2])/(sum(dic[i][3])+0.00001)
			if p==0 or r==0:
				f1=0
			else:
				f1=(2*p*r)/(p+r)


			print(i,p,r,f1,dic[i][4])
def compute_f1_shuxing(file1,file2):
	f1=open(file1,'r')
	f2=open(file2,'r')
	lines1=f1.readlines()
	lines2=f2.readlines()
	f1.close()
	f2.close()
	assert len(lines1)==len(lines2)
	p_list=[]
	r_list=[]
	len_p_list=[]
	len_r_list=[]
	cnt=0
	dic={}
	for i in range(len(lines1)):
		line1=lines1[i].strip().split()
		#print(line1)
		line2=lines2[i].strip().split()
		#print(line2)
		if line1==line2:
			f1=1.0
		#res1=get_entity(line1)
		#res2=get_entity(line2)
		res1=get_multi_entity(line1)
		res2=get_multi_entity(line2)
		temp=p_r_f1_for_shuxing(res1,res2)
		#print(temp)
		for k,v in temp.items():
			if k not in dic:
				dic[k]=v
			else:
				for j in range(len(v)):
					dic[k][j]+=v[j]

		#break

	for k,v in dic.items():
		if v[1]==0 or v[0]==0:
			f1=0.0
		else:
			p=(v[0]+0.0)/v[1]
			r=(v[0]+0.0)/v[3]
			f1=2*p*r/(p+r)
		v.append(f1)
		# print(k,v[0],v[1],v[3],f1)
	dic=sorted(dic.items(),key=lambda k:k[1][-1])
	print(dic)
	for i in dic:
		print(i[0],end=' ')
		for j in i[1]:
			print(j,end=' ')
		print('')


def oov(input1,input2):
	dic=[]
	f=open(input1,'r')
	for line in  f.readlines():
		line=list(line.strip())
		for word in line:
			if word not in dic:
				dic.append(word)
	print(len(dic))
	f.close()
	temp=[]
	f=open(input2,'r')
	for line in  f.readlines():
		line=list(line.strip())
		for word in line:
			if word not in dic:
				temp.append(word)
	print(len(temp),len(set(temp)))
	f.close()



if __name__ == "__main__":
	file1='./result/multi_long_0615_lstmcrf_bio_pred.txt'
	file2='./result/multi_long_0615_lstmcrf_bio_targ.txt'
	# compute_f1_shuxing(file1,file2)
	# compute_f1_shuxinglen(file1,file2)
	# compute_f1_2(file1,file2)
	input1='./data0520/new_x_train.txt'
	input2='./data0520/new_x_test.txt'
	oov(input1,input2)

