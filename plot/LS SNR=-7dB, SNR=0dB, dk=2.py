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

plt.xlabel("用户数 K",font1)
plt.ylabel("和速率",font1)


template_list = [['+','blue'],['v','green'],['*','m'],['D','red']]

x_label_list = [8,10,12,14,16]
ezf_list = [10.26582,10.02692,9.61217,9.05200,8.45807]
duu_MISO_list = [12.56406,13.31149,13.86715,14.24130,14.62841]
learn_UW_list = [11.63176,12.01019,12.14816,12.65290,12.58137]

rwmmse_list = [10.55928,10.53920,10.35930,10.12669,9.87811]

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

plt.axis([8,16,5,15])
scenario_name = 'LS SNR=-7dB, SNR=0dB, d=2'
plt.title(scenario_name,fontt)
save_name = './' + 'LS_SNR_-7dB_SNR_0dB_dk_2'
svg_filepath = save_name+'.svg'
emf_filepath = save_name + '.emf'
pdf_filepath = save_name + '.pdf'
plt.savefig(svg_filepath) 
plt.savefig(pdf_filepath)

inkscape_path = "D:\soft\inkscape\inkscape.exe"
subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
plt.show()



