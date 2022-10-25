import glob
import gzip

print "BLEU non-ensemble"
print "sentBLEU non-ensemble"
print "NIST non-ensemble"

files = glob.glob("../final-metric-scores/baselines/*sys.score.gz")
for f in files:
  lines = [line.rstrip('\n') for line in gzip.open(f)]
  l = lines[0]
  c = l.split()
  print c[0]+" non-ensemble" # all baselines are non-ensemble

files = glob.glob("../final-metric-scores/submissions-processed/*sys.score.gz")

for f in files:

  lines = [line.rstrip('\n') for line in gzip.open(f)]

  #if lines[0].find("ensemble")>-1:
#  print lines[0]

  l = lines[0]

  c = l.split()

  if len(c) > 5:
    if c[5]=="no" or c[5]=="non-ensamble" or c[5]=="non-ensemble":
      c[5] = "non-ensemble"
    print c[0]+" "+c[5]
 
