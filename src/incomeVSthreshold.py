import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

y = []
yp = []
with open('y_y_pred.txt', 'r') as f:
    for line in f:
        y.append(float(line.split(' ')[0]))
        yp.append(float(line.split(' ')[1].strip()))


CVR = 0.17        # Conversion Rate
CPA = 5./1000.    # Cost Per Ad
IPR = 5.          # Income Per buy
revenue = []
threshold = np.linspace(0, 1, 1000)
for CTR_min in threshold:
    revenue.append(sum([IPR*CTR*CVR-CPA  for CTR, y_real in zip(yp, y) if CTR >= CTR_min]))
ctr_best = threshold[np.array(revenue).argmax()]
test_revenue = sum([IPR*y_real*CVR-CPA  for CTR, y_real in zip(yp, y) if CTR >= ctr_best])

a = np.array(y)
baseline = sum(5.*a*CVR)-CPA*len(y) # nomes multipliquen els positus!
print baseline
#####

revenue_new = []
threshold_new = np.linspace(0, 1, 1000)
for CTR_min in threshold_new:
    revenue_new.append(sum([IPR*y_real*CVR-CPA  for CTR, y_real in zip(yp, y) if CTR >= CTR_min]))
ctr_best_new = threshold[np.array(revenue_new).argmax()]


#####

plt.rcdefaults()
params = {'text.usetex': True,
          'text.latex.preamble':[r'\usepackage{txfonts}']}
plt.rcParams.update(params)


with PdfPages('incomeVSthreshold.pdf') as pdf:
    plt.plot(threshold, revenue, color='k')
    plt.xlabel('CTR threshold', fontsize=13)
    plt.ylabel('Predicted revenue [euros]', fontsize=13)
    plt.xlim((0, 0.1))
    plt.annotate(r'Predicted revenue = $\sum_{\mathrm{CTR_{threshold}}}^{\mathrm{CTR_{max}}}(\mathrm{IPR}\times\mathrm{CTR}\times\mathrm{CVR}-\mathrm{CPA})$',
                 xy=(0.95, 0.9), xycoords='axes fraction', fontsize=13,
                 horizontalalignment='right', verticalalignment='bottom')

    plt.annotate('IPR (Income Per buy) = {} euros\nCTR (Click Through rate) in [{:.2f}, {:.2f}]\nCVR (Conversion Rate) = {}\nCPA (Cost Per Ad) = {} euros/ad'.format(IPR, min(yp), max(yp), CVR, CPA),
                 xy=(0.45, 0.7), xycoords='axes fraction', fontsize=14,
                 horizontalalignment='left', verticalalignment='bottom')

    plt.axvline(ctr_best, color='red', linestyle='--')
    plt.annotate('The ad should be shown for users with CTR above {:.3f}\nThe predicted revenue is {:.2f} euros'.format(ctr_best, max(revenue))+'\n'+r'Real revenue = $\sum_{\mathrm{CTR} = 0.006}^{\mathrm{CTR_{max}}}(\mathrm{IPR}\times\mathrm{RealClickLabel}\times\mathrm{CVR}-\mathrm{CPA})$='+'{:.2f} euros'.format(test_revenue),
                 xy=(ctr_best+0.001, 10), color='red', fontsize=13)
    plt.xticks(list(plt.xticks()[0]) + [0] + ctr_best)
    plt.title('Predicted revenue of the CTR prediction model on accomodation Ads\n({} test entries)'.format(len(y)))
    pdf.savefig()

    #######

    plt.clf()
    plt.plot(threshold_new, revenue_new, color='k')
    plt.xlabel('CTR threshold', fontsize=13)
    plt.ylabel('Real revenue [euros]', fontsize=13)
    plt.xlim((0, 0.1))
    plt.annotate(r'Real revenue = $\sum_{\mathrm{CTR_{threshold}}}^{\mathrm{CTR_{max}}}(\mathrm{IPR}\times\mathrm{RealClickLabel}\times\mathrm{CVR}-\mathrm{CPA})$',
                 xy=(0.95, 0.9), xycoords='axes fraction', fontsize=13,
                 horizontalalignment='right', verticalalignment='bottom')

    plt.annotate('IPR (Income Per buy) = {} euros\nCTR (Click Through rate) in [{:.2f}, {:.2f}]\nCVR (Conversion Rate) = {}\nCPA (Cost Per Ad) = {} euros/ad'.format(IPR, min(yp), max(yp), CVR, CPA),
                 xy=(0.45, 0.7), xycoords='axes fraction', fontsize=14,
                 horizontalalignment='left', verticalalignment='bottom')

    plt.axvline(ctr_best_new, color='red', linestyle='--')
    plt.annotate('The ad should be shown for users with CTR above {:.3f}\nThe real revenue is {:.2f} euros'.format(ctr_best_new, max(revenue_new)),
                 xy=(ctr_best_new+0.001, -40), color='red', fontsize=13)
    plt.xticks(list(plt.xticks()[0]) + [0] + ctr_best_new)
    plt.title('Real revenue of the CTR prediction model on accomodation Ads\n({} test entries)'.format(len(y)))
    pdf.savefig()


with open('thresholdVSrevenue.txt', 'wb') as f:
    for i, _ in enumerate(revenue_new):
        f.write(str(threshold_new[i])+' '+str(revenue_new[i])+'\n')

