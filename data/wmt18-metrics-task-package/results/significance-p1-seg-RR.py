import glob
import gzip

submissions = ["../final-metric-scores/baselines/*.seg.score.gz",
		"../final-metric-scores/submissions-processed/*.seg.score.gz"]

f = "../manual-evaluation/RR-seglevel.csv"
lines = [line.rstrip('\n') for line in open(f)]
lines.pop(0)

manual = {}

for l in lines:
  l = l.replace("nmt-smt-hybrid","nmt-smt-hybr")

  c = l.split()

  if len(c) != 5:
    print "error in manual evaluation file"
    exit(1)

  #  en-cs newstest2017 1200 LIUM-FNMT.4852 tuning-task-afrl_4gb.sgm.0
  lp = c[0]
  data = c[1]
  sid = c[2] 
  better = c[3]
  worse = c[4]

  # remove dedup score
#  c = system.split("+")
#  system = c[0]

  if lp not in manual:
    manual[lp] = {}
  if sid not in manual[lp]:
    manual[lp][sid] = {}
  if better not in manual[lp][sid]:
    manual[lp][sid][better] = {}
  if worse not in manual[lp][sid][better]:
    manual[lp][sid][better][worse] = 1
  
missing = 0

met_names = {}
metrics = {}
#lms = {}
#lsm = {}

for s in submissions:
  files = glob.glob(s)
  for f in files:

    lines = [line.rstrip('\n') for line in gzip.open(f)]

    # Various name changes:
    for l in lines:
      l = l.replace("nmt-smt-hybrid","nmt-smt-hybr")
                    
      if (l.find("hybrid")==-1) and (l.find("himl")==-1):

        c = l.split()

        #if ((len(c) != 5) and len(c)!=7) and (len(c)!=9):
        if len(c) < 6:
          missing = missing + 1
        else: # BEER  en-cs himltest2017a Chimera 1 0.509761870499595 no  https://github.com/stanojevic/beer
          metric = c[0] 

          lp = c[1]
          data = c[2]
          system = c[3]
          sid = c[4]
          score = float(c[5]) 

      #    system = system+"::"+sid

          if data != "newstest2018":
            print "error with data set for metric: "+l
            exit(1)

          if lp not in metrics:
            metrics[lp] = {}
          if metric not in metrics[lp]:
            metrics[lp][metric] = {}
          if sid not in metrics[lp][metric]:
            metrics[lp][metric][sid] = {}
          if system not in metrics[lp][metric][sid]:
            metrics[lp][metric][sid][system] = score
    
# check which metrics have scores for all segs    
for lp in manual:
  for metric in metrics[lp]:

    allthere = True

    # check if all manual segments are present for this metric
    for sid in manual[lp]: #[sid][better][worse]
       if not sid in metrics[lp][metric]:
         allthere = False
         print "a) Missing "+lp+" "+metric+" "+sid
       else:
         for s1 in manual[lp][sid]:
           if not s1 in metrics[lp][metric][sid]:
             allthere = False
             print "b) Missing "+lp+" "+metric+" "+sid+" "+s1
           for s2 in manual[lp][sid][s1]:
             if not s2 in metrics[lp][metric][sid]:
               allthere = False
               print "c) Missing "+lp+" "+metric+" "+sid+" "+s1+" "+s2

    if allthere:
      if lp not in met_names:
        met_names[lp] = {}
      if metric not in met_names[lp]:
        met_names[lp][metric] = 1

    #else:
    #  print "segment mismatch "+lp+" "+metric
    #  print sorted(lms[lp][metric])
    #  print sorted(manual[lp])


#for lp in met_names:
#  for system in met_names[lp]:
#    for metric in met_names[lp][system]:
#      print lp+" "+system+" "+metric+" "+met_names[lp][system][metric]

for lp in manual:

  l = lp.replace("-","")

  f = "out/RR-newstest2018-"+l+"-seg-nohy-agree.csv"
  F = open(f,'w')

  s = "SID BETTER WORSE"
  #for metric in sorted(met_names[lp]):
  #    s = s+" "+metric
  #s = s+"\n"

  #for system in manual[lp]:
  #  s = s+lp+" "+system+" "+manual[lp][system]
  #  for metric in sorted(met_names[lp]):
  #    s = s +" "+lsm[lp][system][metric]
  #  s = s+"\n"

  for metric in met_names[lp]:

    s = s + " "+metric

#    conc = 0
#    disc = 0
  s = s + "\n"

  for sid in manual[lp]:
 
    for better in manual[lp][sid]:
      for worse in manual[lp][sid][better]:

        s = s+sid+" "+better+" "+worse
 
        for metric in met_names[lp]:

          if better not in metrics[lp][metric][sid]:
            print "error "+lp+" "+metric+" "+better

          score1 = metrics[lp][metric][sid][better]
          score2 = metrics[lp][metric][sid][worse]

          answer = "0"
  
          if score1 > score2:
            answer = "1"


          s = s + " "+answer  

        s = s +"\n"

  F.write(s)
  F.close()





