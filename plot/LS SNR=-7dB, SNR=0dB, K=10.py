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

plt.xlabel("数据流 d",font1)
plt.ylabel("和速率",font1)


template_list = [['+','blue'],['v','green'],['*','m'],['D','red']]

x_label_list = [1,2,3,4]
ezf_list = [12.31344,10.02692,7.53525,5.47050]
duu_MISO_list = [13.04334,13.31149,13.24629,13.18464]
learn_UW_list = [12.51037,12.01019,12.42555,12.56452]

rwmmse_list = [12.44862,10.53920,8.57527,7.60237]

scheme_list = [ezf_list,rwmmse_list,learn_UW_list,duu_MISO_list]

for i in range(len(scheme_list)):
    plt.plot(x_label_list,scheme_list[i],marker = template_list[i][0],color=template_list[i][1])

legends = ['EZF','WMMSE','LUW','LCP(proposed)']


plt.legend(legends,loc='lower left')
plt.grid()
ax.xaxis.grid(True, which='minor',linewidth=0.2) 
ax.xaxis.grid(True, which='major',linewidth=0.6) 
ax.yaxis.grid(True, which='minor',linewidth=0.2) 
ax.yaxis.grid(True, which='major',linewidth=0.6)

plt.xticks(x_label_list)

plt.axis([1,4,5,15])
scenario_name = 'LS SNR=-7dB, SNR=0dB, K=10'
plt.title(scenario_name,fontt)
save_name = './' + 'LS_SNR_-7dB_SNR_0dB_K_10'
svg_filepath = save_name+'.svg'
emf_filepath = save_name + '.emf'
pdf_filepath = save_name + '.pdf'
plt.savefig(svg_filepath) 
plt.savefig(pdf_filepath)

inkscape_path = "D:\soft\inkscape\inkscape.exe"
subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
plt.show()



