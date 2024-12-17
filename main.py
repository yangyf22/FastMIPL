import os
import time
import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from utils import device, logging, seed_everything, args
from model import FastMIPL
from dataloader import load_data_mat, load_idx_mat, MIPLDataset, create_bags, mil_collate_fn

GLOBAL_WORKER_ID = None

# Weights of dynamic disambiguation
def adjust_alpha(epochs):
    alpha_list = [1.0] * epochs
    for ep in range(epochs):
        alpha_list[ep] = (epochs - ep) / (epochs)
    return alpha_list


# Load data
data_path = os.path.join(args.data_path, args.ds)
index_path = os.path.join(data_path, args.index)
mat_name = args.ds + "_" + args.ds_suffix + ".mat"
logging.info('MAT File Name: {}'.format(mat_name))
mat_path = os.path.join(data_path, mat_name)
ds_name = mat_name[0:-4]
all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab, partial_bag_lab_processed = load_data_mat(
    mat_path, args.nr_fea, args.nr_class, normalize=args.normalize)

# Set the seed for reproducibility
g = torch.Generator()
g.manual_seed(0)
def worker_init_fn(worker_id, seed_value = args.seed):
    seed_everything(seed_value + worker_id)




if __name__ == "__main__":
    if args.cuda:
        logging.info('\tGPU is available!')
    else:
        logging.info('\tGPU is unavailable!')
    
    
    if args.smoke_test:
        all_folds = ['index1.mat']
    else:
        all_folds = ['index1.mat', 'index2.mat', 'index3.mat', 'index4.mat', 'index5.mat',
            'index6.mat', 'index7.mat', 'index8.mat', 'index9.mat', 'index10.mat']
    seed_everything(args.seed) 
    num_fold = len(all_folds)
    start_time = time.time()
    logging.info("\t================================ START ========================================")
    num_trial = args.nr_trial
    alpha_list = adjust_alpha(args.epochs)
    accuracy = np.empty((num_trial, num_fold))
    
    for trial_i in range(num_trial):
        num_i = 0
        for fold_i in all_folds:
            logging.info('\t----------------time: {}, fold: {} ----------------'.format(trial_i, fold_i))
            idx_file = os.path.join(index_path, fold_i)

            # Load the Index and Dataset
            idx_tr, idx_te = load_idx_mat(idx_file)

            # Generate Bags based on Index and Dataset
            x_tr, y_ins_tr, s_bag_tr, y_bag_tr = create_bags(all_ins_fea, bag_idx_of_ins, 
                                                            dummy_ins_lab, bag_lab, 
                                                            partial_bag_lab_processed, 
                                                            idx_tr)
            x_te, y_ins_te, s_bag_te, y_bag_te = create_bags(all_ins_fea, bag_idx_of_ins, 
                                                            dummy_ins_lab, bag_lab, 
                                                            partial_bag_lab_processed, 
                                                            idx_te)
            cov_b = torch.ones((len(x_tr), 1))

            # Initial Model
            model = FastMIPL.initialize_model(x_tr, cov_b, s_bag_tr, args)

            # Ensure Variablie in the Same Device and Convert to Float32
            model.to(device)
            x_tr, x_te, y_ins_tr, s_bag_tr, y_bag_tr = \
                [i.to(device).to(torch.float32) for i in x_tr], [i.to(device).to(torch.float32) for i in x_te], \
                y_ins_tr.to(device).to(torch.float32), s_bag_tr.to(device).to(torch.float32), y_bag_tr.to(device).to(torch.float32)
            cov_b = cov_b.to(device).to(torch.float32)
            
            # Generate DataLoader
            mipl_dataset_tr = MIPLDataset(x_tr, s_bag_tr, y_bag_tr, cov_b) 
            train_loader = DataLoader(mipl_dataset_tr, batch_size=args.bs, shuffle=True, 
                                    collate_fn=None if torch.is_tensor(x_tr) else mil_collate_fn, 
                                    generator=g)
            num_bags_tr = mipl_dataset_tr.__len__()

            # Model Training
            model.train(train_loader, num_bags_tr, weight_list=alpha_list)

            # Model Prediction
            if args.cuda:
                y_pred = model.predict(x_te).cpu().numpy()
            else:
                y_pred = model.predict(x_te).numpy()

            # Evaluate Performance
            acc_te = accuracy_score(y_bag_te, y_pred)
            logging.info('test_acc: {:.3f}'.format(acc_te))
            accuracy[trial_i, num_i] = acc_te
            num_i += 1

            # Clear Memory
            torch.cuda.empty_cache()
            model = None
        logging.info("The mean and std of accuracy at {}-th times {} folds: {}, {}".format(
            trial_i+1, num_i, np.around(np.mean(accuracy[trial_i, :]), 3), 
            np.around(np.std(accuracy[trial_i, :]), 3)))

    logging.info("\t================================ END ========================================")
    end_time = time.time()
    logging.info('The mean and std of accuracy at {} times {} folds: {}, {}'.format(
            num_trial, num_i, np.around(np.mean(accuracy), 3), np.around(np.std(accuracy), 3)))
    logging.info('\tRunning time is {} seconds.'.format(end_time - start_time))
    logging.info('Training is finished.')