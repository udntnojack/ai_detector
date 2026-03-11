import scipy.stats._distn_infrastructure as di
print(di.__file__)
lines=open(di.__file__).read().splitlines()
for i in range(350,380):
    print(i+1, lines[i])
