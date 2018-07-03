




def isNStraightHand(graph):
        """
        :type hand: List[int]
        :type W: int
        :rtype: bool
        """
        def extend_houxuan(houxuan,graph):
            l=[]
            for i in range(len(houxuan)):
                l.append(len(set(houxuan[i])))
            max_l=max(l)

            new_h=[]
            for i in range(len(houxuan)):
                if l[i]<max_l-1:
                    continue
                node=houxuan[i][-1]
                for j in range(len(graph[node])):
                    new_houxuan=houxuan[i][:]
                    new_houxuan.append(graph[node][j])
                    new_h.append(new_houxuan)
            del houxuan
            return new_h
        def judge(houxuan,n):
            for i in range(len(houxuan)):
                h=set(houxuan[i])
                if len(h)==n:
                    return True,len(houxuan[i]),houxuan[i]
            return False,0,[]


        n=len(graph)
        l_list=[]
        for node in range(n):
            houxuan=[[node]]
            Flag=True
            while(Flag):
                houxuan=extend_houxuan(houxuan,graph)
                f,l,r=judge(houxuan,n)
                if f==True:
                    # print(r)
                    l_list.append(l)
                    Flag=True
                    break
        return min(l_list)-1
                



        




        
        


graph=[[7],[3],[3,9],[1,2,4,5,7,11],[3],[3],[9],[3,10,8,0],[7],[11,6,2],[7],[3,9]]
flag=isNStraightHand(graph)
print(flag)