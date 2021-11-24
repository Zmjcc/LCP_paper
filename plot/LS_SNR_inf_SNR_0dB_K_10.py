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

plt.xlabel("data stream d",fontt)
plt.ylabel("sum rate",fontt)


template_list = [['+','blue'],['v','green'],['*','m'],['D','red']]


#OFDM
x_label_list = [1,2,3,4]
ezf_list = [35.27728,37.50172,25.85088,10.31891]
duu_MISO_list = [35.99682,43.51910,44.55647,44.60260]
dl_list = [36.03657,43.35686,44.17270,44.17050]
rwmmse_list = [36.68111,44.25195,45.03310,45.05592]

scheme_list = [ezf_list,rwmmse_list,duu_MISO_list,dl_list]

for i in range(len(scheme_list)):
    plt.plot(x_label_list,scheme_list[i],marker = template_list[i][0],color=template_list[i][1],linestyle = '-')

legends = ['EZF','WMMSE','DUU_MISO_no_net','DUU_MISO_net']

#GS
ezf_list = [14.61397,15.09182,13.39678,10.72613]
duu_MISO_list = [15.96615,17.12452,17.20848,17.21013]
dl_list = [14.80132,14.30294,14.25812,14.22513]
rwmmse_list = [18.05956,19.83158,19.88862,19.88916]


scheme_list = [ezf_list,rwmmse_list,duu_MISO_list,dl_list]

for i in range(len(scheme_list)):
    plt.plot(x_label_list,scheme_list[i],marker = template_list[i][0],color=template_list[i][1],linestyle = '--')

legends = ['EZF','WMMSE','DUU_MISO_no_net','DUU_MISO_net','EZF_GS','WMMSE_GS','DUU_MISO_no_net_GS','DUU_MISO_net_GS']



plt.legend(legends,loc='center right')
plt.grid()
ax.xaxis.grid(True, which='minor',linewidth=0.2) 
ax.xaxis.grid(True, which='major',linewidth=0.6) 
ax.yaxis.grid(True, which='minor',linewidth=0.2) 
ax.yaxis.grid(True, which='major',linewidth=0.6)

plt.xticks(x_label_list)

plt.axis([1,4,10,50])
scenario_name = 'LS SNR=inf, SNR=0dB, K=10'
plt.title(scenario_name,fontt)
#save_name = './'+scenario_name
save_name = './' + 'LS_SNR_inf_SNR_0dB_K_10'
svg_filepath = save_name+'.svg'
emf_filepath = save_name + '.emf'
pdf_filepath = save_name + '.pdf'
plt.savefig(svg_filepath) 
plt.savefig(pdf_filepath)
inkscape_path = "D:\soft\inkscape\inkscape.exe"

plt.show()



