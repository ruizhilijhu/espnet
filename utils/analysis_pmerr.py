
import matplotlib.pyplot as plt
import numpy as np

dir = '/Users/ben_work/data'

file_map = {
'expt_bnfenc_err_cer_10.txt':'bnfenc_cer_bceloss (test)',
'expt_bnfenc_err_wer_10.txt':'bnfenc_wer_bceloss (test)',
'expt_bnfenc_err_cer_11.txt':'bnfenc_cer_mseloss (test)',
'expt_bnfenc_err_wer_11.txt':'bnfenc_wer_mseloss (test)',
'expt_fbank_err_cer_10.txt':'fbank_cer_bceloss (test)',
'expt_fbank_err_wer_10.txt':'fbank_wer_bceloss (test)',
'expt_fbank_err_cer_11.txt':'fbank_cer_mseloss (test)',
'expt_fbank_err_wer_11.txt':'fbank_wer_mseloss (test)',

'expt_bnfenc_err_cer_10.trn.txt':'bnfenc_cer_bceloss (train)',
'expt_bnfenc_err_wer_10.trn.txt':'bnfenc_wer_bceloss (train)',
'expt_bnfenc_err_cer_11.trn.txt':'bnfenc_cer_mseloss (train)',
'expt_bnfenc_err_wer_11.trn.txt':'bnfenc_wer_mseloss (train)',
'expt_fbank_err_cer_10.trn.txt':'fbank_cer_bceloss (train)',
'expt_fbank_err_wer_10.trn.txt':'fbank_wer_bceloss (train)',
'expt_fbank_err_cer_11.trn.txt':'fbank_cer_mseloss (train)',
'expt_fbank_err_wer_11.trn.txt':'fbank_wer_mseloss (train)',

'expt_ctcpresm_err_cer_10.txt':'ctcpresoftmax_cer_bceloss (test)',
'expt_ctcpresm_err_wer_10.txt':'ctcpresoftmax_wer_bceloss (test)',

'expt_bnfenc_err_3class_10.txt':'bnfenc_3class',
'expt_bnfenc_err_3class_10.trn.txt':'bnfenc_3class (train)',


}



####################### utils functions ###################################################
# aurura 8 real and simu wsj 9
def read_result(f, dataset='all', mode='regression', datamode='trn'):
    names = []
    errs = []
    rec_errs = []
    with open(f) as fh:
        for line in fh.readlines():
            name, err, rec_err = line.strip().split()



            if dataset == 'wsj' and len(name) != 8 and datamode=='test':
                continue
            elif dataset == 'chime4real' and not name.endswith('REAL'):
                continue # apply to both test and trn
            elif dataset == 'chime4simu' and not name.endswith('SIMU'):
                continue # apply to both test and trn
            elif dataset == 'aurora4' and len(name) != 9 and datamode=='test':
                continue
            elif dataset == 'aurora4' and not name.endswith('1') and datamode=='trn':
                continue
            elif dataset == 'wsj' and not name.endswith('0') and datamode=='trn':
                continue
            elif dataset == 'dirhaSim' and not name.startswith('Sim'):
                continue
            elif dataset == 'dirhaReal' and not name.startswith('Real'):
                continue

            names.append(name)
            errs.append(float(err))
            if mode=='regression':
                rec_errs.append(float(rec_err[1:-1]))
            elif mode== 'classification':
                rec_errs.append(float(rec_err))

        return names, np.array(errs), np.array(rec_errs)



# ####################### plot the CER and WER histogram for training data ############################
# _, wer, _ = read_result('{}/{}'.format(dir, 'expt_bnfenc_err_wer_10.trn.txt'), dataset='noclean')
# _, cer, _ = read_result('{}/{}'.format(dir, 'expt_bnfenc_err_cer_10.trn.txt'), dataset='noclean')
#
# # TODO: Very unbalance data
#
# plt.figure()
# n, bins, patches = plt.hist(wer, 200, range=[0, 2], density=True)
# # plt.axis([0, 2, 0, 6300])
# plt.xlabel('Groundtruth WER')
# plt.ylabel('Probability %')
# plt.title('Histogram of WER in train data')
# plt.grid(True)
#
# plt.figure()
# n, bins, patches = plt.hist(cer, 200,range=[0, 1], density=True)
# # plt.axis([0, 2, 0, 5])
# plt.xlabel('Groundtruth CER')
# plt.ylabel('Probability %')
# plt.title('Histogram of CER in train data')
# plt.grid(True)
#
# plt.show()


# ####################### regression ############################
markersize=4
err='cer'
feat='bnfenc'

#### test data ######
plt.figure()
id = 'expt_{}_err_{}_10.txt'.format(feat, err)
names, errs, rec_errs = read_result('{}/{}'.format(dir, id), dataset='chime4real', datamode='test')
plt.plot(errs, rec_errs, 'r.', markersize=markersize)
names, errs, rec_errs = read_result('{}/{}'.format(dir, id), dataset='chime4simu', datamode='test')
plt.plot(errs, rec_errs, 'g.', markersize=markersize)

names, errs, rec_errs = read_result('{}/{}'.format(dir, id), dataset='aurora4', datamode='test')
plt.plot(errs, rec_errs, 'c.', markersize=markersize)
names, errs, rec_errs = read_result('{}/{}'.format(dir, id), dataset='wsj', datamode='test')
plt.plot(errs, rec_errs, 'b.', markersize=markersize)

# names, errs, rec_errs = read_result('{}/expt_{}_err_{}_10.dirha.{}.txt'.format(dir, feat, err, 'LA6'), dataset='dirhaReal', datamode='test')
# plt.plot(errs, rec_errs, 'm.', markersize=markersize)
# names, errs, rec_errs = read_result('{}/expt_{}_err_{}_10.dirha.{}.txt'.format(dir, feat, err, 'KA6'), dataset='dirhaReal', datamode='test')
# plt.plot(errs, rec_errs, 'k.', markersize=markersize)

# names, errs, rec_errs = read_result('{}/expt_{}_err_{}_10.dirha.txt'.format(dir, feat, err), dataset='dirhaReal', datamode='test')
# plt.plot(errs, rec_errs, 'm.', markersize=markersize)
# names, errs, rec_errs = read_result('{}/expt_{}_err_{}_10.dirha.txt'.format(dir, feat, err), dataset='dirhaSim', datamode='test')
# plt.plot(errs, rec_errs, 'g.', markersize=markersize)

plt.ylim([0.,1.])
plt.xlim([-0.05,1.1]) if err=='cer' else plt.xlim([-0.1,2])
plt.xlabel('Groundtruth')
plt.ylabel('Prediction')
plt.legend(['chime4real', 'chime4simu', 'aurora4','wsj', 'dirhaRealLA6', 'dirhaRealKA6'])
# plt.legend(['dirhaReal', 'dirhaSim'])

plt.title('{}'.format(file_map[id]))


# #### train data ######
# plt.figure()
# id = 'expt_{}_err_{}_10.trn.txt'.format(feat, err)
# names, errs, rec_errs = read_result('{}/{}'.format(dir, id), dataset='chime4real')
# plt.plot(errs, rec_errs, 'r.', markersize=markersize)
# names, errs, rec_errs = read_result('{}/{}'.format(dir, id), dataset='chime4simu')
# plt.plot(errs, rec_errs, 'g.', markersize=markersize)
#
# names, errs, rec_errs = read_result('{}/{}'.format(dir, id), dataset='aurora4')
# plt.plot(errs, rec_errs, 'c.', markersize=markersize)
# names, errs, rec_errs = read_result('{}/{}'.format(dir, id), dataset='wsj')
# plt.plot(errs, rec_errs, 'b.', markersize=markersize)
#
# plt.ylim([0.,1.])
# plt.xlim([-0.05,1.1]) if err=='cer' else plt.xlim([-0.1,2])
# plt.xlabel('Groundtruth')
# plt.ylabel('Prediction')
# plt.legend(['chime4real', 'chime4simu', 'aurora4','wsj'])
# plt.title('{}'.format(file_map[id]))
plt.show()


#
#
#
#
#
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
