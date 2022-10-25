import gzip
import glob
import sys

met = sys.argv[1]

in_dir  = './'
out_dir  = './chrF-process/'
segF = met+'.seg.score.gz'
hybridsD = '../../creating-hybrids/hybrid-descriptions/newstest2018-hybrids' 
testset = 'newstest2018'

o = gzip.open(out_dir + met + '.hybrid.sys.score.gz', 'wb')

#BEER cs-en newstest2018 CUNI-Transformer.5560 18 0.5372341955559389 non-ensamble https://github.com/stanojevic/beer
print 'Reading metric segment scores...',
mt2scoreD = {}
with gzip.open(in_dir + segF, 'rb') as f:
    for lines in f.readlines():
	parts = lines.rstrip().split()
	lan, ts, mt, seg_id, score = parts[1], parts[2], parts[3], parts[4], parts[5]
	if not ts == testset: continue
	kk = mt + '.' + lan + '-' + seg_id 
	
	if mt2scoreD.has_key(kk):
	    print 'Error!'
	else:
	    mt2scoreD[kk] = float(score)

print 'DONE'

def read_mtsystem_key(filepath):
    mt_keyD = {}

    lines = open(filepath)
    next(lines)
    for ll in lines:
	[kk, mt_system] = ll.strip().split()
	mt_keyD[kk] = mt_system

    return mt_keyD

def compute_lan_hybrids(filepath):
    mt_keyD = read_mtsystem_key(filepath)

    hybridF = filepath[:-len('.system-key')]
    
    lines = open(hybridF)
    next(lines)

    for ll in lines: # for each hybrid file
	score = 0.
	parts = ll.strip().split()
	lan, idx, keyL = parts[0], parts[1], parts[2:]

	for ii, kk in enumerate(keyL): # for each line in the hybrid file
	    mt = mt_keyD[kk]
	    score += mt2scoreD[mt + '.' + lan + '-' + str(ii+1)]

	hybrid_score = score / (ii+1)

	o.write(met + "\t" + lan + "\t" + testset + "\thybrid." + idx + "\t" + str(hybrid_score) + "\n")

print 'Computing hybrids scores...'
for filepath in glob.glob(hybridsD + '/*.system-key'):  
    lan = filepath.split('/')[-1].split('.')[-3]
    print filepath
    compute_lan_hybrids(filepath)
    '''if met == 'BLEND' and lan == 'en-ru': 
	print filepath	
	compute_lan_hybrids(filepath)
    elif met == 'YiSi-1_srl' and lan in ['en-zh', 'en-de']:
	print filepath
	compute_lan_hybrids(filepath) 
    elif (met in['BLEND', 'meteor++', 'RUSE', 'UHH_TSKM', 'YiSi-1_srl']) and lan.startswith('en-'): 
	pass
    else:
	print filepath	
	compute_lan_hybrids(filepath)
    '''

o.close()
print 'DONE'
