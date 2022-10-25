import glob
import gzip

submissions = ["../final-metric-scores/baselines/*.sys.score.gz",
		"../final-metric-scores/submissions-processed/*.sys.score.gz"]

f = "../manual-evaluation/DA-syslevel.csv"
lines = [line.rstrip('\n') for line in open(f)]
lines.pop(0)

manual = {}

for l in lines:
  l = l.replace("nmt-smt-hybrid","nmt-smt-hybr")

  c = l.split()

  if len(c) != 3:
    print "erorr in manual evaluation file"
    exit(1)
 
  lp = c[0]
  score = c[1]
  system = c[2] 

  if lp not in manual:
    manual[lp] = {}
  if system not in manual[lp]:
    manual[lp][system] = score
  
missing = 0

met_names = {}
lms = {}
lsm = {}

for s in submissions:
  files = glob.glob(s)
  for f in files:

    lines = [line.rstrip('\n') for line in gzip.open(f)]

    for l in lines:
      l = l.replace("nmt-smt-hybrid","nmt-smt-hybr")
                    
      if (l.find("hybrid")==-1) and (l.find("himl")==-1):

        c = l.split()

        if ((len(c) != 5) and len(c)!=7) and (len(c)!=9):
          missing = missing + 1

        else:
          metric = c[0] 
          lp = c[1]
          data = c[2]
          system = c[3]
          score = c[4] 

          if data != "newstest2018":
            print "error with data set for metric: "+l
            exit(1)

          if lp not in lms:
            lms[lp] = {}
          if metric not in lms[lp]:
            lms[lp][metric] = {}
          if system not in lms[lp][metric]:
            lms[lp][metric][system] = score
          
          if lp not in lsm:
            lsm[lp] = {}
          if system not in lsm[lp]:
            lsm[lp][system] = {}
          if system not in lsm[lp][system]:
            lsm[lp][system][metric] = score
    
# check which metrics have scores for all systems       
for lp in manual:
  for metric in lms[lp]:
    if sorted(lms[lp][metric])==sorted(manual[lp]):
  
      if lp not in met_names:
        met_names[lp] = {}
      if metric not in met_names[lp]:
        met_names[lp][metric] = 1

    else:
      print "systems mismatch "+lp+" "+metric
      print sorted(lms[lp][metric])
      print sorted(manual[lp])


#for lp in met_names:
#  for system in met_names[lp]:
#    for metric in met_names[lp][system]:
#      print lp+" "+system+" "+metric+" "+met_names[lp][system][metric]

for lp in manual:

  l = lp.replace("-","")

  f = "out/DA-newstest2018-"+l+"-sys-nohy-scores.csv"
  F = open(f,'w')

  s = "LP SYSTEM HUMAN"

  #for metric in sorted(lms[lp]):
  for metric in sorted(met_names[lp]):
  #for metric in sorted(met_names[lp]["online-B.0"]):
      s = s+" "+metric

  s = s+"\n"

  for system in manual[lp]:
    s = s+lp+" "+system+" "+manual[lp][system]

    for metric in sorted(met_names[lp]):

      s = s +" "+lsm[lp][system][metric]
    s = s+"\n"

  F.write(s)
  F.close()





