import sys
import glob 
import os.path

DATA = sys.argv[1]
HYBRID = sys.argv[2]
LEVEL = sys.argv[3]

nonensemble = {}

f = "out/metrics-ensemble.csv"
lines = [line.rstrip('\n') for line in open(f)]
for l in lines:
  c = l.split()
  metric = c[0]
  ensem = c[1]
  
  if ensem == "non-ensemble":
    nonensemble[metric] = 1

f = "out/*-"+DATA+"*en-*"+LEVEL+"*"+HYBRID+"*cor.csv"
toen = glob.glob(f)
f = "out/*-"+DATA+"*-en*"+LEVEL+"*"+HYBRID+"*cor.csv"
ento = glob.glob(f)

files = sorted(toen)+sorted(ento)



def printTable( files):
  r  = {}
  names = {}
  systems = {}
  winners = {}

  for f in files:

    print "%% "+f

    if f.find("en-")> -1:
      i = f.find("en-")-2
      lp = f[i:i+4]
    else:
      i = f.find("-en")+1
      lp = f[i:i+4]

    winnerf = f.replace("cor.csv","winners.csv")
    if os.path.exists(winnerf): 
      lines = [line.rstrip('\n') for line in open(winnerf)]

      for l in lines:
        c = l.split()
        metric = c[0]
        winner = c[1]
        if lp not in winners:
          winners[lp] = {}
        if metric not in winners[lp]:
          winners[lp][metric] = winner

    lines = [line.rstrip('\n') for line in open(f)]

    for l in lines:
      c = l.split()

      metric = c[0]
      cor = c[1]
      n = c[2]

      if metric not in names:
        names[metric] = 1
  
      if lp not in r:
        r[lp] = {}
      if metric not in r[lp]:
        if HYBRID == "hybrids":
          r[lp][metric] = round(float(cor),4)
        else:
          r[lp][metric] = round(float(cor),3)
      if lp not in systems:
        systems[lp] = n

  s = "                           "
  for lp in sorted(r):  
    l = lp[0:2]+"-"+lp[2:4]
    s = s+" & {\\bf "+l+"}" 
  s = s + " \\\\[1ex]\n"
  s = s+"                           "
  for lp in sorted(r): 
    n = str(systems[lp])
    while len(n) < 10:
      n = " "+n 
    s = s+" & "+n+" " 
  s = s + " \\\\[1ex]\n"
  
  for metric in sorted(names, key=lambda s: s.lower()):
    m = metric
    m = m.replace('YiSi.',"YiSi-")
    m = m.replace('meteor..',"meteor++")
    m = m.replace('YiSi_',"YiSi-")
    m = m.replace("chrF.","chrF+")
    #del names[metric]
    #names[m] = 1
    if m in nonensemble:
      s = s+"\\nonen "  
    else:
      s = s+"\\ensem "  


    m = m.replace("_","\\_")
    m = "\\metric{"+m+"} "
    while len(m) < 26:
      m = m+" "

    s = s+m+" "
    for lp in sorted(r):
      if metric in r[lp]:
        correl = str(r[lp][metric])
        while len(correl) < 5:
          correl = correl+"0"

        if lp in winners and winners[lp][metric] == "YES":
          s = s + " & {\\bf "+correl+"}"
        else:
          s = s + " &      "+correl+" "

      else:
        s = s + " &         $-$"

    s = s + " \\\\\n" 
  print s    

printTable(toen)
print ""
printTable(ento)
