import sys
import os.path
F=sys.argv[1]

#f = "out/metric-names.csv"
#lines = [line.rstrip('\n') for line in open(f)]
#metrics = {}

#for l in lines:
#  c = l.split()  
#  metrics[c[1]] = c[0]

#files = [F+"-cor.csv",F+"-sig.csv"]
files = [ F+"-bootstrap.csv"]

for f in files:

  if os.path.exists(f): 
    lines = [line.rstrip('\n') for line in open(f)]
    N = open(f,'w')

    for l in lines:
      #newl = l.replace("_DOT_",".")
      #newl = newl.replace("_PLUS_","+")
      #newl = newl.replace("_DASH_","-")
      newl = l.replace('chrF.', 'chrF+')
      newl = newl.replace('meteor..', 'meteor++')
      newl = newl.replace('YiSi.', 'YiSi-')

      N.write(newl+"\n")

    N.close()
