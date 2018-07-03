# -*- coding:utf-8 -*-  


filex='new_x_test.txt'
filey='new_y_test.txt'

f=open('data.txt','w')
fx=open(filex,'r')
linex=fx.readlines()
fx.close()
fy=open(filey,'r')
liney=fy.readlines()
fy.close()

for i in range(len(linex)):
	i=59
	linex[i]=linex[i].replace('　','#')
	lx=linex[i].strip().split(' ')
	ly=liney[i].strip().split(' ')
	assert len(lx)==len(ly)
	for j in range(len(ly)):
		f.write(lx[j]+' '+ly[j]+'\n')
	f.write('-----------------------------------------------\n')
	break
f.close()
