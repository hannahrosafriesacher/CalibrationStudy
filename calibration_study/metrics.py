import torch
import torchmetrics
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy

import scipy.stats as sci
import numpy as np
import matplotlib.pyplot as plt
import wandb

#Classification: Expected Calibration Error (ECE)
class ECE(Metric):
    def __init__(self, bins):
        super().__init__()
        self.add_state("predictions", [], dist_reduce_fx = "cat")
        self.add_state("true_labels", [], dist_reduce_fx = "cat")
        self.bins = bins
        self.bounds=torch.arange(0,1,1/self.bins)

    #Split input array according to predictions in equally spaced bins (same bin width)
    def split_arrays_ECE(self, predictions, true_labels):
        nr_bounds=torch.arange(0,len(self.bounds))
        true_labels_list = []
        predictions_list = []
        bin_sizes_list = []
        pos_ratio_list = []
        nr_pos_list = []
        nr_neg_list = []

        mask = np.digitize(predictions, self.bounds)
        for index in nr_bounds:
            index = index.item()
            true_labels_list.append(true_labels[mask == index+1])
            predictions_list.append(predictions[mask == index+1])
            bin_sizes_list.append(torch.tensor(true_labels_list[-1].shape[0]))
            pos_ratio_list.append((true_labels_list[-1] == 1).sum() / true_labels_list[-1].shape[0])
            nr_pos_list.append((true_labels_list[-1] == 1).sum())
            nr_neg_list.append((true_labels_list[-1] != 1).sum())
        return true_labels_list, predictions_list, bin_sizes_list, pos_ratio_list, nr_pos_list, nr_neg_list
  
    #Calculate Calibration Error (Nixon et al.: https://doi.org/10.48550/arXiv.1904.01685):
    def calculate_error(self, posRatio_bin, meanProb_bin, bin_size):
        return (torch.abs(posRatio_bin - meanProb_bin) * bin_size)
        #               |    acc(b)    -    conf(n)  | * nb

    #Calculate variance estimate by sum of variances of every of Beta distribution
    def estimate_variance(self, input_arr):
        p = (input_arr == 1).sum()
        n = input_arr.shape[0] - p
        return (n*p)/((n+p)*(n+p+1)) 

    #Calculate ECE-metric for current Batch
    def update(self, predictions, true_labels):
        self.predictions.append(predictions)
        self.true_labels.append(true_labels)
        
    def compute(self):
        
        # TODO CHECK with batching for valid and test!
        #assert len(self.predictions) > 1, 'ECE computed on a single batch'
        
        ECE_true_labels, ECE_predictions, ECE_bin_sizes, ECE_pos_ratios,_ ,_ = self.split_arrays_ECE(torch.cat(self.predictions), torch.cat(self.true_labels))
        ece = 0
        total = 0
        for bin_ind in list(range(self.bins)):
            #For current bin...
            ECE_collector = ECE_true_labels[bin_ind], ECE_predictions[bin_ind], ECE_bin_sizes[bin_ind], ECE_pos_ratios[bin_ind]
            ece_bin = 0
            #...calculate ECE:
            if ECE_collector[2] != 0:
                #...obtain positive ratio (=acc calculated from true values) for current bin if bin has measurements
                posRatio_ECE = ECE_collector[3]
                #...obtain probablity mean (=conf calculated from predictions) for each split for current bin if bin has measurements
                meanProb_ECE = torch.mean(ECE_collector[1].float())
                #...calculate error for current bin and weigh it according to bin size:
                ece_bin = self.calculate_error(posRatio_bin=posRatio_ECE, meanProb_bin=meanProb_ECE, bin_size=ECE_collector[2])
            else:
                pass
            #Obtain ECE for current batch
            ece += ece_bin
            total += ECE_collector[2].item()
        if total == 0:
            return torch.tensor(0)
        #Compute total ECE over all batches
        return ece/total

    def plot(self, wandb_logger):
        stats_box=[]
        total = []
        mean_pred = []
        pos_ratios = []
        _, pred, _, ratios, nr_pos, nr_neg = self.split_arrays_ECE(torch.cat(self.predictions), torch.cat(self.true_labels))
        for bin_index in range(self.bins):
            Stats = {}
            pos=nr_pos[bin_index].cpu().numpy()
            neg=nr_neg[bin_index].cpu().numpy()
            
            if pos == 0 and neg !=0 :
                me = 0
            elif pos != 0 and neg == 0:
                me = 1
            else:
                me = sci.beta.mean(pos ,neg)
            med = sci.beta.median(pos ,neg)
            q1, q3 = sci.beta.interval(0.5, pos ,neg )
            whislo, whishi = sci.beta.interval(0.95, pos ,neg)
            box_data_bin = [('med', med), ('me', me), ('q1', q1), ('q3', q3), ('whislo', whislo), ('whishi', whishi)]
            total.append(pos + neg)
            
            Stats.update(box_data_bin)
            stats_box.append(Stats)
            mean_pred.append(torch.mean(pred[bin_index]).cpu().numpy())
            pos_ratios.append(ratios[bin_index].cpu().numpy())
            

        x_labels=[str(self.bounds.cpu().numpy()[x]) + '-' + str(self.bounds.cpu().numpy()[x + 1]) + '\n' + '#' + str(total[x]) for x in range(self.bins)]
        medianprops = dict(linestyle='--',  color='black')
        fig, axs=plt.subplots()
        axs.bxp(stats_box, showmeans=False, showfliers=False,medianprops= medianprops, positions=np.arange(0.5, 10.5, 1.0).tolist())
        axs.set_xticklabels(x_labels, rotation = 45, size=8)
        axs.plot(np.arange(0.5, 10.5, 1.0).tolist(), np.array(pos_ratios), '*', c='black')
        axs.set_ylim(-0.05,1.05)
        axs.stairs(mean_pred, np.arange(0.0, 11, 1.0).tolist(), fill=False, color='firebrick', linestyle='--')
         
        axs.set_xlabel('Predicted activity', size=16)
        axs.set_ylabel('Positive ratio', size=16)

        wandb_logger.log({"Classification Calibration (Equal Width)": wandb.Image(plt)})

        

#Classification: Adaptive Calibration Error (ACE)
class ACE(Metric):
    def __init__(self, bins):
        super().__init__()
        self.add_state("predictions", [], dist_reduce_fx = "cat")
        self.add_state("true_labels", [], dist_reduce_fx = "cat")
        self.bins = bins
        self.bounds = torch.arange(0,1+1/self.bins,1/self.bins)
        
        
    #Split input array according to predictions in equally sized bins (varying bin width)
    def split_arrays_ACE(self, predictions, true_labels):
        bin_sizes_list = []
        pos_ratio_list = []
        nr_pos_list = []
        nr_neg_list = []
        sorted_index = torch.argsort(predictions)
        true_labels_split = torch.tensor_split(true_labels[sorted_index], self.bins)
        predictions_split = torch.tensor_split(predictions[sorted_index], self.bins)

        
        for index in list(range(self.bins)):
            bin_sizes_list.append(torch.tensor(true_labels_split[index].shape[0]))
            pos_ratio_list.append((true_labels_split[index] == 1).sum() / true_labels_split[index].shape[0])
            nr_pos_list.append((true_labels_split[index] == 1).sum())
            nr_neg_list.append((true_labels_split[index] != 1).sum())
        return true_labels_split, predictions_split, bin_sizes_list, pos_ratio_list, nr_pos_list, nr_neg_list

        
    #calculate Calibration Error (Nixon et al.: https://doi.org/10.48550/arXiv.1904.01685):
    def calculate_error(self, posRatio_bin, meanProb_bin, bin_size):
        return (torch.abs(posRatio_bin - meanProb_bin) * bin_size)
        #               |    acc(b)    -    conf(n)  | * nb

    #Calculate variance estimate by sum of variances of every of Beta distribution
    def estimate_variance(self, input_arr):
        p = (input_arr==1).sum()
        n = input_arr.shape[0] - p
        return (n*p)/((n+p)*(n+p+1)) 

    def update(self, predictions, true_labels):
        self.predictions.append(predictions)
        self.true_labels.append(true_labels)

    def compute(self):
        ACE_true_labels, ACE_predictions, ACE_bin_sizes, ACE_pos_ratios, _, _=self.split_arrays_ACE(torch.cat(self.predictions), torch.cat(self.true_labels))
        ace = 0
        total = 0
        for bin_ind in list(range(self.bins)):
            #For current bin...
            ACE_collector = ACE_true_labels[bin_ind], ACE_predictions[bin_ind], ACE_bin_sizes[bin_ind], ACE_pos_ratios[bin_ind]
            ace_bin = 0
            #...calculate ACE:
            if ACE_collector[2] != 0:
                #...obtain positive ratio (=acc calculated from true values) for current bin if bin has measurements
                posRatio_ACE = ACE_collector[3]
                #...obtain probablity mean (=conf calculated from predictions) for each split for current bin if bin has measurements
                meanProb_ACE = torch.mean(ACE_collector[1].float())
                #...calculate error for current bin and weigh it according to bin size:
                ace_bin += self.calculate_error(posRatio_bin=posRatio_ACE, meanProb_bin=meanProb_ACE, bin_size=ACE_collector[2])
                
            else:
                pass
        #Obtain ACE for current batch
            ace += ace_bin  
            total += ACE_collector[2]
            
        #Compute total ACE, over all batches
        return ace/total


class BrierScore(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("bs", torch.tensor(0.0), dist_reduce_fx = "sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx = "sum")

    def update(self, predictions, true_labels):
        bs_batch = torch.square(predictions - true_labels).sum()

        self.bs += bs_batch
        self.total += predictions.shape[0]
        
    def compute(self):
        bs = self.bs/self.total
        return bs
    
class Refinement(Metric):
    def __init__(self, bins):
        super().__init__()
        self.add_state("predictions", [], dist_reduce_fx = "cat")
        self.add_state("true_labels", [], dist_reduce_fx = "cat")
        self.bins = bins
        self.bounds=torch.arange(0,1,1/self.bins)
          
    #Split input array according to predictions in equally spaced bins (same bin width)
    def split_arrays_ECE(self, predictions, true_labels):
        nr_bounds=torch.arange(0,len(self.bounds))
        true_labels_list = []
        predictions_list = []
        bin_sizes_list = []
        pos_ratio_list = []
        nr_pos_list = []
        nr_neg_list = []

        mask = np.digitize(predictions, self.bounds)
        for index in nr_bounds:
            index = index.item()
            true_labels_list.append(true_labels[mask == index+1])
            predictions_list.append(predictions[mask == index+1])
            bin_sizes_list.append(torch.tensor(true_labels_list[-1].shape[0]))
            pos_ratio_list.append((true_labels_list[-1] == 1).sum() / true_labels_list[-1].shape[0])
            nr_pos_list.append((true_labels_list[-1] == 1).sum())
            nr_neg_list.append((true_labels_list[-1] != 1).sum())
        return true_labels_list, predictions_list, bin_sizes_list, pos_ratio_list, nr_pos_list, nr_neg_list

    def update(self, predictions, true_labels):
        self.predictions.append(predictions)
        self.true_labels.append(true_labels)
        
    def compute(self):
        rm = 0
        total = 0
        ECE_true_labels, ECE_predictions, ECE_bin_sizes, ECE_pos_ratios,_ ,_ = self.split_arrays_ECE(torch.cat(self.predictions), torch.cat(self.true_labels))
        for bin_ind in range(self.bins):
            ECE_collector = ECE_true_labels[bin_ind], ECE_predictions[bin_ind], ECE_bin_sizes[bin_ind], ECE_pos_ratios[bin_ind]
            if ECE_collector[2] != 0:
                rm_bin = ECE_collector[2] * (ECE_collector[3] * (1 - ECE_collector[3]))
                total += ECE_collector[2]
                rm += rm_bin
            else:
                pass
        return rm/total

class Reliability(Metric):
    def __init__(self, bins):
        super().__init__()
        self.add_state("predictions", [], dist_reduce_fx = "cat")
        self.add_state("true_labels", [], dist_reduce_fx = "cat")
        self.bins = bins
        self.bounds=torch.arange(0,1,1/self.bins)

    #Split input array according to predictions in equally spaced bins (same bin width)
    def split_arrays_ECE(self, predictions, true_labels):
        nr_bounds=torch.arange(0,len(self.bounds))
        true_labels_list = []
        predictions_list = []
        bin_sizes_list = []
        pos_ratio_list = []
        nr_pos_list = []
        nr_neg_list = []

        mask = np.digitize(predictions, self.bounds)
        for index in nr_bounds:
            index = index.item()
            true_labels_list.append(true_labels[mask == index+1])
            predictions_list.append(predictions[mask == index+1])
            bin_sizes_list.append(torch.tensor(true_labels_list[-1].shape[0]))
            pos_ratio_list.append((true_labels_list[-1] == 1).sum() / true_labels_list[-1].shape[0])
            nr_pos_list.append((true_labels_list[-1] == 1).sum())
            nr_neg_list.append((true_labels_list[-1] != 1).sum())
        return true_labels_list, predictions_list, bin_sizes_list, pos_ratio_list, nr_pos_list, nr_neg_list

    def update(self, predictions, true_labels):
        self.predictions.append(predictions)
        self.true_labels.append(true_labels)
        
    def compute(self):
        rl = 0
        total = 0
        ECE_true_labels, ECE_predictions, ECE_bin_sizes, ECE_pos_ratios,_ ,_ = self.split_arrays_ECE(torch.cat(self.predictions), torch.cat(self.true_labels))
        for bin_ind in range(self.bins):
            ECE_collector = ECE_true_labels[bin_ind], ECE_predictions[bin_ind], ECE_bin_sizes[bin_ind], ECE_pos_ratios[bin_ind]
            pred_mean = torch.mean(ECE_collector[1].float())
            if ECE_collector[2] != 0:
                rl_bin = ECE_collector[2] * torch.square(pred_mean - ECE_collector[3])
                total += ECE_collector[2]
                rl += rl_bin
            else:
                pass
        return rl/total
