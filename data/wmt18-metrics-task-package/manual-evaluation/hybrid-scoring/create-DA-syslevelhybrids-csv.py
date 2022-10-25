import glob

DATA = "newstest2018"

files = glob.glob("../../../wmt18-sys-test/proc-hits-combo/analysis/ad-seg-scores-*.csv")

print "LP DATA HYBRID HUMAN"

def get_score(DA_score,line,systems):

  c = line.split()
  
  lp = c[0]
  hybrid = c[1]
  seg = 1
  count = 0
  total = 0

  for i in xrange(2,len(c)):

    system_key = c[i]
    system_name = systems[system_key]
   
    if system_name in DA:
      if seg in DA[system_name]:  
#        print lp+" "+hybrid+" "+system_key+" "+system_name+" "+DA_score[system_name][str(seg)]

        total = total + DA_score[system_name][seg]
        count = count + 1
    else:
      print "no "+system_name 
      exit(1)

    seg = seg + 1
 
  score = total/count

  return score

for f in files:

  DA = {}

  lines = [line.rstrip('\n') for line in open(f)]
  lines.pop(0)

  for l in lines:
    c = l.split() 
    syst = c[0]
    sid = c[1]
    z_scr = c[3]

    if syst not in DA:
      DA[syst] = {}
    elif sid not in DA[syst]:
      DA[syst][int(sid)] = float(z_scr)
    else:
      print "error system and sid already exist "+syst+" "+sid+" "+z_scr
   
  lp = f.replace("../../../wmt18-sys-test/proc-hits-combo/analysis/ad-seg-scores-","") 
  lp = lp.replace(".csv","")
  src = lp[0:2]
  trg = lp[3:]
  lp = src+"-"+trg

  hybrid_file = "../../creating-hybrids/hybrid-descriptions/"+DATA+"-hybrids/"+DATA+"."+lp+".hybrids"
  hybrid_key_file = "../../creating-hybrids/hybrid-descriptions/"+DATA+"-hybrids/"+DATA+"."+lp+".hybrids.system-key"

  lines = [line.rstrip('\n') for line in open(hybrid_key_file)] 
  lines.pop(0)

  system = {}

  for l in lines:

    c = l.split()

    letter = c[0]
    name = c[1]

    system[letter] = name

  lines = [line.rstrip('\n') for line in open(hybrid_file)] 
  lines.pop(0)
  
  for HY in xrange(0,10000):
    hybrid_DA_score = get_score(DA,lines[HY],system)
    print src+"-"+trg+" "+DATA+" "+str(HY)+" "+str(hybrid_DA_score)



