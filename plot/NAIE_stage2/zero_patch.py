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

x_label_list = [8,9,10,11,12]
ezf_list = [25.00002,24.32109,23.34873,21.74378,21.77050]
lcp_universal_list = [31.46025,32.84438,34.02925,35.00220,35.45073]
lcp_specified_list = [31.80906,33.03837,34.31569,35.31842,36.27900]
wmmse_list = [34.87080,36.59645,38.17042,39.62756,40.88282]
scheme_list = [ezf_list,wmmse_list,lcp_specified_list,lcp_universal_list]

for i in range(len(scheme_list)):
    plt.plot(x_label_list,scheme_list[i],marker = template_list[i][0],color=template_list[i][1])

legends = ['EZF','WMMSE','LCP','LCP_zero_patch']


plt.legend(legends,loc='lower left')
plt.grid()
ax.xaxis.grid(True, which='minor',linewidth=0.2) 
ax.xaxis.grid(True, which='major',linewidth=0.6) 
ax.yaxis.grid(True, which='minor',linewidth=0.2) 
ax.yaxis.grid(True, which='major',linewidth=0.6)

plt.xticks(x_label_list)

plt.axis([8,12,20,45])
scenario_name = 'LS SNR=-7dB, SNR=0dB, K=12, zero patch'
plt.title(scenario_name,fontt)
save_name = './' + 'zero_patch'
svg_filepath = save_name+'.svg'
emf_filepath = save_name + '.emf'
pdf_filepath = save_name + '.pdf'
plt.savefig(svg_filepath) 
plt.savefig(pdf_filepath)

inkscape_path = "D:\soft\inkscape\inkscape.exe"
#subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
plt.show()



