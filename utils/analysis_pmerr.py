
import matplotlib.pyplot as plt
import numpy as np

dir = '/Users/ben_work/data'


####################### utils functions ###################################################
# aurura 8 real and simu wsj 9
def read_result(f, dataset='all', mode='regression', datamode='trn'):
    names = []
    errs = []
    rec_errs = []
    with open(f) as fh:
        for line in fh.readlines():
            name, err, rec_err = line.strip().split()

            if dataset == 'wsj' and not name.endswith('0') and datamode=='train':
                continue
            elif dataset == 'wsj' and len(name) != 8 and datamode=='test':
                continue
            elif dataset == 'aurora4' and not name.endswith('1') and datamode=='train':
                continue
            elif dataset == 'aurora4' and len(name) != 9 and datamode=='test':
                continue
            elif dataset == 'chime4real' and not name.endswith('REAL'):
                continue # apply to both test and trn
            elif dataset == 'chime4simu' and not name.endswith('SIMU'):
                continue # apply to both test and trn
            elif dataset == 'dirhaSim' and not name.startswith('Sim'):
                continue
            elif dataset == 'dirhaReal' and not name.startswith('Real'):
                continue
            elif not (dataset in name):
                continue


            names.append(name)
            errs.append(float(err))
            if mode=='regression':
                rec_errs.append(float(rec_err[1:-1]))
            elif mode== 'classification':
                rec_errs.append(float(rec_err))

        return names, np.array(errs), np.array(rec_errs)

def plot_pmerr(dir, id, dataset, datamode, color, markersize=3):
    names, errs, rec_errs = read_result('{}/{}'.format(dir, id), dataset=dataset, datamode=datamode)
    rho = np.corrcoef(errs, rec_errs)
    plt.plot(errs, rec_errs, color, markersize=markersize, label='{:.3f}, {}'.format(rho[0,1], dataset))
    return dataset, rho[0,1]


# # ####################### plot the CER and WER histogram for training test data ############################
# _, wer, _ = read_result('{}/{}'.format(dir, 'expt_bnfenc_err_wer_10.test.txt'), dataset='noclean')
# _, cer, _ = read_result('{}/{}'.format(dir, 'expt_bnfenc_err_cer_10.test.txt'), dataset='noclean')
#
# # TODO: Very unbalance data
#
# plt.figure()
# n, bins, patches = plt.hist(wer, 200, range=[0, 2], density=True)
# # plt.axis([0, 2, 0, 6300])
# plt.xlabel('Groundtruth WER')
# plt.ylabel('Probability %')
# plt.title('Histogram of WER in PM test data')
# plt.savefig('{}/{}.png'.format('/Users/ben_work/Desktop/plots', 'hist_pm_test_wer'), format='png',dpi=300)
# plt.grid(True)
#
# plt.figure()
# n, bins, patches = plt.hist(cer, 200,range=[0, 1], density=True)
# # plt.axis([0, 2, 0, 5])
# plt.xlabel('Groundtruth CER')
# plt.ylabel('Probability %')
# plt.title('Histogram of CER in PM test data')
# plt.savefig('{}/{}.png'.format('/Users/ben_work/Desktop/plots', 'hist_pm_test_cer'), format='png',dpi=300)
# plt.grid(True)
#
# plt.show()


# ####################### regression ############################
# markersize=3
# err='cer'
# feat='bnfenc'
# id_tmp = 'expt_{}_err_{}_20.{}.txt'
# loss='bceloss'
#
# # #### train data ######
# rho_list = []
# plt.figure()
# id = id_tmp.format(feat, err, 'train')
# rho_list.append(plot_pmerr(dir, id, 'aurora4', 'train', 'c.', markersize))
# rho_list.append(plot_pmerr(dir, id, 'wsj', 'train', 'b.', markersize))
# rho_list.append(plot_pmerr(dir, id, 'chime4real', 'train', 'r.', markersize))
# rho_list.append(plot_pmerr(dir, id, 'chime4simu', 'train', 'g.', markersize))
#
# plt.ylim([0.,1.]); plt.xlim([-0.05,1.1]) if err=='cer' else plt.xlim([-0.1,2])
# plt.xlabel('Groundtruth'); plt.ylabel('Prediction')
# plt.legend(loc='upper left')
# name = '{}_{}_{} ({})'.format(feat, err, loss, 'train')
# plt.title(name)
# plt.savefig('{}/{}.png'.format('/Users/ben_work/Desktop/plots',name), format='png', dpi=300)
#
# # print corelation coefficient
# print (name)
# for i,j in rho_list:
#     print "{:.3f}".format(j)
#
#
# #### test data ######
# rho_list = []
# plt.figure()
# id = id_tmp.format(feat, err, 'test')
# rho_list.append(plot_pmerr(dir, id, 'wsj', 'test', 'b.', markersize))
# rho_list.append(plot_pmerr(dir, id, 'aurora4', 'test', 'c.', markersize))
# rho_list.append(plot_pmerr(dir, id, 'chime4real', 'test', 'r.', markersize))
# rho_list.append(plot_pmerr(dir, id, 'chime4simu', 'test', 'g.', markersize))
# rho_list.append(plot_pmerr(dir, id, 'dirhaReal', 'test', 'm.'))
# rho_list.append(plot_pmerr(dir, id, 'dirhaSim', 'test', 'k.'))
#
# plt.ylim([0.,1.]); plt.xlim([-0.05,1.1]) if err=='cer' else plt.xlim([-0.1,2])
# plt.xlabel('Groundtruth'); plt.ylabel('Prediction')
# plt.legend(loc='upper left')
# name = '{}_{}_{} ({})'.format(feat, err, loss, 'test')
# plt.title(name)
# plt.savefig('{}/{}.png'.format('/Users/ben_work/Desktop/plots',name), format='png',dpi=300)
# print (name)
# for i,j in rho_list:
#     print "{:.3f}".format(j)



####################### speaker  ############################
markersize=3
err='cer'
feat='bnfenc'
id_tmp = 'expt_{}_err_{}_20.{}.txt'
loss='bceloss'


# #### train data ######
rho_list = []
plt.figure()
id = id_tmp.format(feat, err, 'train')
rho_list.append(plot_pmerr(dir, id, 'BUS.CH', 'train', 'c.', markersize))

plt.ylim([0.,1.]); plt.xlim([-0.05,1.1]) if err=='cer' else plt.xlim([-0.1,2])
plt.xlabel('Groundtruth'); plt.ylabel('Prediction')
plt.legend(loc='upper left')
name = '{}_{}_{} ({})'.format(feat, err, loss, 'train')
plt.title(name)
# plt.savefig('{}/{}.png'.format('/Users/ben_work/Desktop/plots',name), format='png', dpi=300)

# print corelation coefficient
print (name)
for i,j in rho_list:
    print "{:.3f}".format(j)


# #### test data ######
# rho_list = []
# plt.figure()
# id = id_tmp.format(feat, err, 'test')
# rho_list.append(plot_pmerr(dir, id, 'wsj', 'test', 'b.', markersize))
# rho_list.append(plot_pmerr(dir, id, 'aurora4', 'test', 'c.', markersize))
# rho_list.append(plot_pmerr(dir, id, 'chime4real', 'test', 'r.', markersize))
# rho_list.append(plot_pmerr(dir, id, 'chime4simu', 'test', 'g.', markersize))
# rho_list.append(plot_pmerr(dir, id, 'dirhaReal', 'test', 'm.'))
# rho_list.append(plot_pmerr(dir, id, 'dirhaSim', 'test', 'k.'))
#
# plt.ylim([0.,1.]); plt.xlim([-0.05,1.1]) if err=='cer' else plt.xlim([-0.1,2])
# plt.xlabel('Groundtruth'); plt.ylabel('Prediction')
# plt.legend(loc='upper left')
# name = '{}_{}_{} ({})'.format(feat, err, loss, 'test')
# plt.title(name)
# plt.savefig('{}/{}.png'.format('/Users/ben_work/Desktop/plots',name), format='png',dpi=300)
# print (name)
# for i,j in rho_list:
#     print "{:.3f}".format(j)




# # ####################### classification ############################
# from sklearn.metrics import confusion_matrix
# id = 'expt_bnfenc_err_3class_10.trn.txt'
# _, y_true, y_pred = read_result('{}/{}'.format(dir, id), dataset='all', mode='classification', datamode='test')
# cm = confusion_matrix(y_true, y_pred)
# print cm



#
# # ####################### regression ############################
# markersize=1
# err='cer'
#
# dataset='chime4simu'
#
# #### train ######
# plt.figure()
# id = 'expt_{}_err_{}_10.trn.txt'.format('fbank', err)
# names_1, errs_1, rec_errs_1 = read_result('{}/{}'.format(dir, id), dataset=dataset, datamode='trn')
# id = 'expt_{}_err_{}_10.trn.txt'.format('bnfenc', err)
# names_2, errs_2, rec_errs_2 = read_result('{}/{}'.format(dir, id), dataset=dataset, datamode='trn')
# plt.plot(rec_errs_1-errs_1, rec_errs_2-errs_1, '.', markersize=markersize)
# plt.plot([-0.5,0.5],[-0.5,0.5])
# plt.ylim([-0.5,0.5])
# plt.xlim([-0.5,0.5])
#
# plt.grid()
# plt.xlabel('fbank')
# plt.ylabel('bnfenc')
# plt.show()




plt.show()