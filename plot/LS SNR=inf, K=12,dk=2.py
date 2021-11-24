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

plt.xlabel("信噪比 SNR",font1)
plt.ylabel("和速率",font1)


template_list = [['+','blue'],['v','green'],['*','m'],['D','red']]

x_label_list = [-5,0,5,10,15]
ezf_list = [17.32468,37.51610,67.05324,102.55030,140.81165]
duu_MISO_list = [26.07988,46.53421,74.28740,107.68790,144.53784]
learn_UW_list = [23.80750,40.32045,65.50490,96.91969,118.01130]

rwmmse_list = [26.44371,47.64301,76.84924,111.95805,149.62735]

scheme_list = [ezf_list,rwmmse_list,learn_UW_list,duu_MISO_list]

for i in range(len(scheme_list)):
    plt.plot(x_label_list,scheme_list[i],marker = template_list[i][0],color=template_list[i][1])

legends = ['EZF','WMMSE','LUW','LCP(proposed)']


plt.legend(legends,loc='upper left')
plt.grid()
ax.xaxis.grid(True, which='minor',linewidth=0.2) 
ax.xaxis.grid(True, which='major',linewidth=0.6) 
ax.yaxis.grid(True, which='minor',linewidth=0.2) 
ax.yaxis.grid(True, which='major',linewidth=0.6)

plt.xticks(x_label_list)

plt.axis([-5,15,15,150])
scenario_name = 'LS SNR=inf, K=12,d=2'
plt.title(scenario_name,fontt)
save_name = './' + 'LS_SNR_inf_K_12_dk_2'
svg_filepath = save_name+'.svg'
emf_filepath = save_name + '.emf'
pdf_filepath = save_name + '.pdf'
plt.savefig(svg_filepath) 
plt.savefig(pdf_filepath)

inkscape_path = "D:\soft\inkscape\inkscape.exe"
subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
plt.show()



