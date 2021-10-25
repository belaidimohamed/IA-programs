# def classGrouping(levels, maxSpread):
#     levels.sort()
#     group = []
#     for i in range(len(levels)):
#         if levels[i+1]-levels[i] < maxSpread:
#             group.append(levels[i])
#
#     return group
def pthFactor(n, p):
    l = []
    for i in range(1,n+1):
        if len(l) > p:
            break
        if n%i == 0 :
            l.append(i)
    print(l)
    try :
        return l[p-1]
    except:
        return 0

print(pthFactor(55,3))