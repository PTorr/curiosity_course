import numpy as np
import os
import glob


T = 7999

path = '/home/torr/PycharmProjects/curiosity_course/part2_ANN/good examples'
extension = 'csv'
os.chdir(path)
csv_files = [i for i in glob.glob('*.{}'.format(extension))]
hl = {}
alphas = {}
errors = {}
l = 0
for f in csv_files:
    a = f.split('hls')
    c = a[1].split('.')
    c = c[0].split('[')
    c = c[1].split(']')
    c = c[0].split(',')
    r = map(int, c)
    hl[l] = r

    a = a[0].split('learning_rate')
    a = a[1].split(' ')
    a = a[0].replace('_', '.')
    a = a.replace('[', '')
    a = a.replace(']', '')
    a = a.replace(',', '')
    r1 = float(a)
    alphas[l] = r1

    data = np.genfromtxt(f,delimiter=',',skip_header=0)
    errors[l] = data[0]
    l+=1

AIC = {}
BIC = {}
for j in range(l):
    hls = hl[j]
    m = 3*hls[0]+(hls[0]+1)*hls[1]+(hls[1]+1)*hls[2]+(hls[2]+1)*1
    AIC[j] = np.log(2*errors[j]) + 2*m/T
    BIC[j] = np.log(2*errors[j]) + 2*np.log(m)/T

for j in range(l):
    if np.isinf(AIC[j]) == 'True':
        print np.isinf(AIC[j])
        AIC[j] = 0

print AIC.values()
print BIC.values()