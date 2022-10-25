import sys
import os
import glob
import random

SRC = sys.argv[1]
TRG = sys.argv[2]
DATA = sys.argv[3]
H = 10000
D = "../input/wmt18-metrics-task-no-hybrids/wmt18-submitted-data/txt/system-outputs/"+DATA
DIR='hybrid-descriptions/newstest2018-hybrids' # pre: DATA+"-hybrid-pairs"

os.system("mkdir -p "+DIR)

print "Creating hybrids for "+SRC+" "+TRG

submissions = glob.glob(D+"/"+SRC+"-"+TRG+"/*")

N = len(submissions)
M = len([line.rstrip('\n') for line in open(submissions[0])])

systems = []

for subm in submissions:
  M1 = len([line.rstrip('\n') for line in open(subm)])

  if M1!=M:
    print "error: size of submissions files are not consistent: "+str(subm)
    exit(1)

  system = subm.replace(D+"/"+SRC+"-"+TRG+"/"+DATA+".","")
  system = system.replace("."+SRC+"-"+TRG,"")
  systems.append(system)

fn = DIR+"/"+DATA+"."+SRC+"-"+TRG+".hybrids.system-key"
random.shuffle(systems)
O = open(fn,'w')
O.write("KEY MT-SYSTEM\n")
for i in xrange(0,len(systems)):
  O.write(chr(65+i)+"    "+systems[i]+"\n")
O.close()

s = "LP HYBRID"

for i in xrange(1,M+1):
  s += " "+str(i)
s += "\n"

for i in xrange(0,H):
  s += SRC+"-"+TRG+" "+str(i)

  sys1 = ""
  sys2 = ""

  while sys1 == sys2:
    sys1 = chr(random.randint(65,65+(N-1)))
    sys2 = chr(random.randint(65,65+(N-1)))
  
  for j in xrange(1,M+1):
    choice = random.randint(0,1)
    if choice ==0:
      system = sys1
    else:
      system = sys2

    s += " "+system
  s += "\n"

fn = DIR+"/"+DATA+"."+SRC+"-"+TRG+".hybrids"
O = open(fn,'w')
O.write(s)
O.close()

