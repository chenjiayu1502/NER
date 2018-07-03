def compute_label_acc(pred,targ):
        assert len(pred)==len(targ)
        #print(len(pred))
        acc=0.0
        all=0.001
        for i in range(len(pred)):
            print(len(pred[i]))
            assert len(pred[i])==len(targ[i])
            all+=len(pred[i])
            for j in range(len(pred[i])):
                if pred[i][j]==targ[i][j]:
                    acc+=1.0
            print(acc,all)
        return acc/all

pred=['O O O O O B I O O O O O'.split(),
'O O O O O O O O B I I'.split(),
'O O O B I I I I I I I'.split()]
targ=['O O pp O O B I O O y O O'.split(),
'O O O O p O O O B I I'.split(),
'O O O B I I n I o I I'.split()]
acc=compute_label_acc(pred,targ)
print(acc)