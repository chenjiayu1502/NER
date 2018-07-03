#encoding=utf-8


file_x='x_train.txt'
file_y='y_train.txt'
new_x='x_train_0622.txt'
new_y='y_train_0622.txt'

fx=open(file_x,'r')
fy=open(file_y,'r')
linesx=fx.readlines()
linesy=fy.readlines()
fx.close()
fy.close()
fx=open(new_x,'w')
fy=open(new_y,'w')
for i in range(len(linesx)):
	lx=linesx[i].strip().split(' ')
	ly=linesy[i].strip().split(' ')
	nx=[]
	ny=[]
	for j in range(len(lx)):
		s=['　',' ',' ']
		if lx[j] in s:
			continue
		nx.append(lx[j])
		ny.append(ly[j])
	assert(len(nx)==len(ny))
	fx.write(' '.join(nx)+'\n')
	fy.write(' '.join(ny)+'\n')
	