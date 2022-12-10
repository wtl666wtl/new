import matplotlib.pyplot as plt
import math

plt.figure(dpi=150)

dd = {
    'edit_ratio':  [0, 0.0501, 0.1172, 0.2320, 0.3284, 0.4188],
    '1': [0, -0.10, -0.90, -3.05, -6.72, -10.77],
    '2': [0, -0.39, -7.88, -16.62, -30.37, -39.79],
    '3': [0, -11.06, -15.92, -25.57, -37.07, -53.09],
    '4': [0, -1.79, -4.16, -8.58, -13.92, -20.88],
    '5': [0, -2.27, -7.77, -13.77, -20.78, -25.13],
}

DO_LOG = True

plt.plot(dd['edit_ratio'], dd['1'], 'r.-', label = 'BERTScore')
plt.plot(dd['edit_ratio'], dd['4'], '.--', label = 'MoverScore')
plt.plot(dd['edit_ratio'], dd['5'], '.--', label = 'BARTScore')
plt.plot(dd['edit_ratio'], dd['2'], 'g.--', label = 'BLEU')
plt.plot(dd['edit_ratio'], dd['3'], 'b.--', label = 'COMET')



plt.grid('on')

plt.xlabel('noise-ratio', fontsize=13)
plt.ylabel('relative score change', fontsize=13)

plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)

plt.legend(fontsize=13)
plt.savefig('bertS_attack.png')
