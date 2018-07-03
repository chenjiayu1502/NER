#coding=utf-8
import random

def split_data(filex,filey):
	f1=open(filex,'r')
	f2=open(filey,'r')
	lines1=f1.readlines()
	lines2=f2.readlines()
	f1.close()
	f2.close()
	fx_train=open('x_train.txt','w')
	fx_test =open('x_test.txt','w')
	fy_train=open('y_train.txt','w')
	fy_test =open('y_test.txt','w')

	#分割训练集合和测试集合，并以空格分开
	for i in range(len(lines1)):
		linex=list(lines1[i].strip())
		liney=lines2[i].strip().split(',')
		k=random.randint(1,10000)
		if k % 10 == 1:
			fx_test.write(' '.join(linex)+'\n')
			fy_test.write(' '.join(liney)+'\n')
		else:
			fx_train.write(' '.join(linex)+'\n')
			fy_train.write(' '.join(liney)+'\n')

def change_label():
	f=open('y_train.txt','r')
	lines=f.readlines()
	f.close()
	f=open('y_train_bio.txt','w')
	for line in lines:
		line=line.strip().split()
		temp=[]
		for w in line:
			temp.append(w[0])
		f.write(' '.join(temp)+'\n')
	f.close()

def change_o():
	f_in=open('new_y_train.txt','r')
	f_out=open('new_y_train2.txt','w')
	for line in f_in.readlines():
		line=line.strip().split()
		temp=[]
		for l in line:
			if l=='O':
				k=random.randint(0,10)
				temp.append(l+str(k))
			else:
				temp.append(l)
		f_out.write(' '.join(temp)+'\n')
	f_in.close()
	f_out.close()

if __name__ == "__main__":
	change_o()
	# filex='thex.txt'
	# filey='they.txt'
	# #split_data(filex,filey)
	# change_label()
