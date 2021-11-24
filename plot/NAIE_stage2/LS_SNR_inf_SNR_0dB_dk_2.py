import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy import io
import subprocess
inkscape_path = "F:\\soft\\Inkscape\\inkscape.exe"
ax=plt.subplot(111)


font1 = {'family' : 'STSong',
'weight' : 'normal',
'size'   : 20,
}
fontt = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}

plt.xlabel("user K",font1)
plt.ylabel("Sum Rate",font1)


template_list = [['+','blue'],['v','green'],['*','m'],['D','red']]

x_label_list = [8,9,10,11,12,13,14,15,16]
ezf_list = [26.30144,25.49072,24.72096,23.53084,22.33274,20.57807,18.88887,17.01587,17.01587]
duu_MISO_list = [32.97712,34.56903,36.15296,37.55572,38.79725,39.94706,41.00529,42.08843,42.08843]
#learn_UW_list = [11.63176,12.01019,12.14816,12.65290,12.58137]
lcp_list = [31.80906,33.03837,34.31569,35.31842,36.27900,37.94825,38.28802,39.50515,39.50515]
rwmmse_list = [35.04549,36.68260,38.30254,39.75802,41.00999,42.20031,43.31966,44.39919,44.39919]

scheme_list = [ezf_list,rwmmse_list,duu_MISO_list,lcp_list]

for i in range(len(scheme_list)):
    plt.plot(x_label_list,scheme_list[i],marker = template_list[i][0],color=template_list[i][1])

legends = ['EZF','WMMSE','MISO_no_net','LCP(proposed)']


plt.legend(legends,loc='lower left')
plt.grid()
ax.xaxis.grid(True, which='minor',linewidth=0.2) 
ax.xaxis.grid(True, which='major',linewidth=0.6) 
ax.yaxis.grid(True, which='minor',linewidth=0.2) 
ax.yaxis.grid(True, which='major',linewidth=0.6)

plt.xticks(x_label_list)

plt.axis([8,16,15,45])
scenario_name = 'LS SNR=-7dB, SNR=0dB, d=2'
plt.title(scenario_name,fontt)
save_name = './' + 'LS_SNR_-7dB_SNR_0dB_dk_2'
svg_filepath = save_name+'.svg'
emf_filepath = save_name + '.emf'
pdf_filepath = save_name + '.pdf'
plt.savefig(svg_filepath) 
plt.savefig(pdf_filepath)

inkscape_path = "D:\soft\inkscape\inkscape.exe"
#subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
plt.show()



