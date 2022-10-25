import gzip 
import sys

inf = sys.argv[1]
o1 = gzip.open('submissions-processed/'+inf, 'wb')
o2 = gzip.open('submissions-processed-remainder-for-testsuites/'+inf, 'wb')

with gzip.open('submissions-corrected/'+inf, 'r') as f:
    for line in f.readlines():
        parts = line.rstrip().split()
        ts = parts[2]
        if ts == 'newstest2018':
            o1.write(line)
        else:
            o2.write(line)

o1.close()
o2.close()
