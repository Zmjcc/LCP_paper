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

plt.xlabel("user number K",fontt)
plt.ylabel("sum rate",fontt)


template_list = [['+','blue'],['v','green'],['*','m'],['D','red']]

x_label_list = [8,10,12,14,16]
ezf_list = [36.07519,37.50172,37.47161,36.22896,33.92975]
duu_MISO_list = [39.57110,43.51910,46.77262,49.46900,51.82211]
learn_UW_list = [35.24810,38.69106,40.31998,43.50746,46.17963]
dl_list = [39.44856,43.35686,46.47407,49.01803,51.26411]
rwmmse_list = [40.13234,44.25195,47.59734,50.37708,52.79394]

scheme_list = [ezf_list,rwmmse_list,dl_list,duu_MISO_list]

for i in range(len(scheme_list)):
    plt.plot(x_label_list,scheme_list[i],marker = template_list[i][0],color=template_list[i][1])


ezf_list = [15.09604,15.09182,14.72653,14.09396,13.26024]
duu_MISO_list = [16.60372,17.12452,17.48187,17.75469,17.96996]
learn_UW_list = [35.24810,38.69106,40.31998,43.50746,46.17963]
dl_list = [15.03167,14.30294,14.43567,14.65276,14.69571]
rwmmse_list = [19.18336,19.83158,20.25115,20.55836,20.80102]

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

plt.axis([8,16,10,55])
scenario_name = 'LS SNR=inf, SNR=0dB, d=2'
plt.title(scenario_name,fontt)
save_name = './' + 'LS_SNR_inf_SNR_0dB_dk_2'
svg_filepath = save_name+'.svg'
emf_filepath = save_name + '.emf'
pdf_filepath = save_name + '.pdf'
plt.savefig(svg_filepath) 
plt.savefig(pdf_filepath)
inkscape_path = "D:\soft\inkscape\inkscape.exe"
#subprocess.call([inkscape_path, svg_filepath, '--export-emf', emf_filepath])
plt.show()



