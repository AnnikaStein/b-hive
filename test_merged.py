print("start import")
import os
import uproot as u
print("finish import")

listbranch = ['jet_pt', 'jet_eta']

goods = []
with open('training_files.txt','r') as f:
    lines = f.read().splitlines()

for i in lines:
    try:
        u.open(i+":deepntuplizer/tree")
        goods.append(i)
    except:
        print(i)

with open('samples_good.txt', 'w') as fp:
    for item in goods:
        fp.write("%s\n" % item)
    print('Done')
