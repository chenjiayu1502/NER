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
	print(p_all,r_all)
	print(2*p_all*r_all/(p_all+r_all))
	#print(sum(f1_list)/len(f1_list))


if __name__ == "__main__":
	file1='./result/multi_long_0528_lstmcrf_pred.txt'
	file2='./result/multi_long_0528_lstmcrf_targ.txt'
	compute_f1_2(file1,file2)

