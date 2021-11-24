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

x_label_list = [1,2,3,4]
ezf_list = [28.93736,24.23218,14.01567,6.00165]
duu_MISO_list = [31.94044,35.93478,36.54956,36.57140]
wmmse_list = [33.41554,38.09211,38.86535,38.90887]
lcp_list = [30.87823,34.16869,34.08330,33.94239]
rwmmse_list = [12.44862,10.53920,8.57527,7.60237]

scheme_list = [ezf_list,wmmse_list,duu_MISO_list,lcp_list]

for i in range(len(scheme_list)):
    plt.plot(x_label_list,scheme_list[i],marker = template_list[i][0],color=template_list[i][1])

legends = ['EZF','WMMSE','MISO_no_net','LCP(proposed)']


plt.legend(legends,loc='center right')
plt.grid()
ax.xaxis.grid(True, which='minor',linewidth=0.2) 
ax.xaxis.grid(True, which='major',linewidth=0.6) 
ax.yaxis.grid(True, which='minor',linewidth=0.2) 
ax.yaxis.grid(True, which='major',linewidth=0.6)

plt.xticks(x_label_list)

plt.axis([1,4,5,40])
scenario_name = 'LS SNR=infB, SNR=0dB, K=10'
plt.title(scenario_name,fontt)
save_name = './' + 'LS_SNR_inf_SNR_0dB_K_10'
svg_filepath = save_name+'.svg'
emf_filepath = save_name + '.emf'
pdf_filepath = save_name + '.pdf'
plt.savefig(svg_filepath) 
plt.savefig(pdf_filepath)

inkscape_path = "D:\soft\inkscape\inkscape.exe"
subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
plt.show()



