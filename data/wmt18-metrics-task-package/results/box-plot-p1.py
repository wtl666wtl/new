import glob
import sys

DATA = sys.argv[1]

metrics = {}

if DATA != "*":
  f = "out/*"+DATA+"*sys-nohy-cor.csv"
else:
  f = "out/*sys-nohy-cor.csv"

files = glob.glob(f)

for f in files:

  if f.find("XX") < 0:

    lines = [line.rstrip('\n') for line in open(f)]

    for l in lines:
      c = l.split()
      metric = c[0]
   
      if metric not in metrics:
        metrics[c[0]] = 1
      else:
        metrics[metric] = metrics[metric] + 1

valuelist = metrics.values()
valuelist.sort()
highest = valuelist[len(valuelist)-1]

final = []

for m in metrics:  
  if metrics[m] == highest:
    final.append(m)

mets = {}

for m in final:
 # print m+" "+str(metrics[m])
  mets[m] = 1

#print str(highest)

print "METRIC R N"

for f in files:

  #if (f != "out/DA-newstest2017-enzh-sys-nohy-cor.csv") and (f.find("XX")<0):
  if f.find("XX")<0:

    lines = [line.rstrip('\n') for line in open(f)]

    for l in lines:
      c = l.split()

      if c[0] in mets:
        #l=l.replace('YiSi.', 'YiSi-')
        #l=l.replace('chrF.', 'chrF+')
        print l

  
