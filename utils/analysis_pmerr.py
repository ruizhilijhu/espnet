
import matplotlib.pyplot as plt
import numpy as np

dir = '/Users/ben_work/data'

file_map = {
'expt_bnfenc_err_cer_1.txt':'bnfenc_cer_bceloss',
'expt_bnfenc_err_wer_1.txt':'bnfenc_wer_bceloss',
'expt_fbank_err_cer_1.txt':'fbank_cer_bceloss',
'expt_fbank_err_wer_1.txt':'fbank_wer_bceloss',
'expt_bnfenc_err_cer_2.txt':'bnfenc_cer_mseloss',
'expt_bnfenc_err_wer_2.txt':'bnfenc_wer_mseloss',
'expt_bnfenc_err_cer_2_sigmoid_mse.txt':'bnfenc_cer_mseloss_sigmoid_mse',
'expt_bnfenc_err_wer_2_sigmoid_mse.txt':'bnfenc_wer_mseloss_sigmoid_mse',
'expt_bnfenc_err_cer_2_sigmoid_mse.trn.txt':'bnfenc_cer_mseloss_sigmoid_mse (train data)',
'expt_bnfenc_err_wer_2_sigmoid_mse.trn.txt':'bnfenc_wer_mseloss_sigmoid_mse (train data)',
'expt_bnfenc_err_cer_2.trn.txt':'bnfenc_cer_mseloss (train data)',
'expt_bnfenc_err_wer_2.trn.txt':'bnfenc_wer_mseloss (train data)',
'expt_bnfenc_err_cer_1.trn.txt':'bnfenc_cer_bceloss (train data)',
'expt_bnfenc_err_wer_1.trn.txt':'bnfenc_wer_bceloss (train data)',
'expt_fbank_err_cer_1.trn.txt':'fbank_cer_bceloss (train data)',
'expt_fbank_err_wer_1.trn.txt':'fbank_wer_bceloss (train data)',
}



####################### utils functions ###################################################
# aurura 8 real and simu wsj 9
def read_result(f, dataset='all'):
    names = []
    errs = []
    rec_errs = []
    with open(f) as fh:
        for line in fh.readlines():
            name, err, rec_err = line.strip().split()

            if dataset == 'wsj' and len(name) != 8:
                continue
            elif dataset == 'chime4real' and not name.endswith('REAL'):
                continue
            elif dataset == 'chime4simu' and not name.endswith('SIMU'):
                continue
            elif dataset == 'aurora4' and len(name) != 9:
                continue
            elif dataset == 'seen' and len(name) not in [9, 8]:
                continue
            elif dataset == 'unseen' and len(name) in [9, 8]:
                continue
            elif dataset == 'noclean' and  (name.endswith('0') or name.endswith('1')):
                continue

            names.append(name)
            errs.append(float(err))
            rec_errs.append(float(rec_err))

        return names, np.array(errs), np.array(rec_errs)



# ####################### plot the CER and WER histogram for training data ############################
# _, wer, _ = read_result('{}/{}'.format(dir, 'expt_bnfenc_err_wer_1.trn.txt'), dataset='noclean')
# _, cer, _ = read_result('{}/{}'.format(dir, 'expt_bnfenc_err_cer_1.trn.txt'), dataset='noclean')
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




dataset='unseen'
id = 'expt_bnfenc_err_wer_1.trn.txt'
names, errs, rec_errs = read_result('{}/{}'.format(dir, id), dataset=dataset)

plt.figure()
plt.plot(errs, rec_errs, '.')
plt.ylim([0,1])
plt.xlim([0,1])
plt.xlabel('Groundtruth')
plt.ylabel('Prediction')
plt.title('{}, data:{}'.format(file_map[id], dataset))
plt.show()
