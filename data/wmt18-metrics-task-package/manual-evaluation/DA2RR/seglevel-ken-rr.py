import sys

SRC=sys.argv[1]
TRG=sys.argv[2]
DATA=sys.argv[3]

if SRC == "cs":
  print "LP SID BETTER WORSE"

f = "../../../wmt18-sys-test/proc-hits-combo/analysis/ad-seg-scores-"+SRC+"-"+TRG+".csv"

lines = [line.rstrip('\n') for line in open(f)]
lines.pop(0)

scores = {}
              

for l in lines:
  c = l.split()

  # PJATK.4761 1243 89.2142857142857 1.21043349715233 14

  system = c[0] 
  sid = c[1]
  score = float(c[2])

  if system != "HUMAN":
    if sid not in scores:
      scores[sid] = {}

    if system not in scores[sid]:
      scores[sid][system] = score

cnt = 0
sid_cnt = 0
margin = 0
tot = 0

for sid in scores:

  if len(scores[sid]) > 1:


    for s1 in scores[sid]:

      for s2 in scores[sid]:

        if s1 > s2:

          sid_cnt = sid_cnt + 1
          
          scr1 = scores[sid][s1]
          scr2 = scores[sid][s2]

          if abs(scr1-scr2) >=25:
            margin = margin + 1          

            if scr1 > scr2:
              #print SRC+"-"+TRG+" "+sid+" "+s1+","+s2+" "+str(scr1)+" "+str(scr2)
              print SRC+"-"+TRG+" "+DATA+" "+sid+" "+s1+" "+s2
            else:
              #print SRC+"-"+TRG+" "+sid+" "+s2+","+s1+" "+str(scr2)+" "+str(scr1)
              print SRC+"-"+TRG+" "+DATA+" "+sid+" "+s2+" "+s1

for sid in scores:
  if len(scores[sid])>1:
    cnt = cnt + 1
    tot = tot + len(scores[sid])
#len(scores[sid])
  
f = "summary."+SRC+TRG
F = open(f,'w')  
F.write(SRC+"-"+TRG+" DA judgments "+str(cnt)+"\n")  
F.write(SRC+"-"+TRG+" ave DA judgments "+str(float(tot)/cnt)+"\n")  
F.write(SRC+"-"+TRG+" DA combos "+str(sid_cnt)+"\n")  
F.write(SRC+"-"+TRG+" > 25 dist "+str(margin)+"\n")  
F.close()
