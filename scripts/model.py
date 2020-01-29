import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from sklearn.metrics import average_precision_score, roc_auc_score
from evaluation import smin, fmax


def train(device, net, criterion, learning_rate, lr_sched, num_epochs,
          train_loader, train_loader_eval, valid_loader, icvec, ckpt_dir, logs_dir, 
          evaluate_train=True, save_step=10):

    # Define logger
    logger = SummaryWriter(logs_dir)
    # Define optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # Define scheduler for learning rate adjustment
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Load checkpoint model and optimizer
    start_epoch = load_checkpoint(net, optimizer, scheduler, filename=ckpt_dir+'/model_last.pth.tar')

    # Evaluate validation set before start training
    print("[*] Evaluating epoch %d..." % start_epoch)
    avg_valid_loss, avg_valid_avgprec, avg_valid_rocauc, avg_valid_sdmin, avg_valid_fmax, _, _ = evaluate(device, net, criterion, valid_loader, icvec)
    print("--- Average valid loss:                  %.4f" % avg_valid_loss)
    print("--- Average valid avg precision score:   %.4f" % avg_valid_avgprec)
    print("--- Average valid roc auc score:         %.4f" % avg_valid_rocauc)
    print("--- Average valid min semantic distance: %.4f" % avg_valid_sdmin)
    print("--- Average valid max F-score:           %.4f" % avg_valid_fmax)

    # Start training phase
    print("[*] Start training...")
    # Training epochs
    for epoch in range(start_epoch, num_epochs):
        net.train()
        # Print current learning rate
        print("[*] Epoch %d..." % (epoch + 1))
        for param_group in optimizer.param_groups:
            print('--- Current learning rate: ', param_group['lr'])

        for data in train_loader:
            # Get current batch and transfer to device
            data = data.to(device)
            labels = data.y

            with torch.set_grad_enabled(True):  # no need to specify 'requires_grad' in tensors
                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # Forward pass
                _, outputs = net(data)
                current_loss = criterion(outputs, labels)

                # Backward pass and optimize
                current_loss.backward()
                optimizer.step()

        # Save last model
        state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        torch.save(state, ckpt_dir + '/model_last.pth.tar')

        # Save model at epoch
        if (epoch + 1) % save_step == 0:
            print("[*] Saving model epoch %d..." % (epoch + 1))
            torch.save(state, ckpt_dir + '/model_epoch%d.pth.tar' % (epoch + 1))
        
        # Evaluate all training set and validation set at epoch
        print("[*] Evaluating epoch %d..." % (epoch + 1))
        if evaluate_train:
            avg_train_loss, avg_train_avgprec, avg_train_rocauc, avg_train_sdmin, avg_train_fmax, _, _ = evaluate(device, net, criterion, train_loader_eval, icvec)
            print("--- Average train loss:                  %.4f" % avg_train_loss)
            print("--- Average train avg precision score:   %.4f" % avg_train_avgprec)
            print("--- Average train roc auc score:         %.4f" % avg_train_rocauc)
            print("--- Average train min semantic distance: %.4f" % avg_train_sdmin)
            print("--- Average train max F-score:           %.4f" % avg_train_fmax)
            
            logger.add_scalar('train_loss_epoch', avg_train_loss, epoch + 1)
            logger.add_scalar('train_avgprec_epoch', avg_train_avgprec, epoch + 1)
            logger.add_scalar('train_rocauc_epoch', avg_train_rocauc, epoch + 1)
            logger.add_scalar('train_sdmin_epoch', avg_train_sdmin, epoch + 1)
            logger.add_scalar('train_fmax_epoch', avg_train_fmax, epoch + 1)

        avg_valid_loss, avg_valid_avgprec, avg_valid_rocauc, avg_valid_sdmin, avg_valid_fmax, _, _ = evaluate(device, net, criterion, valid_loader, icvec)
        print("--- Average valid loss:                  %.4f" % avg_valid_loss)
        print("--- Average valid avg precision score:   %.4f" % avg_valid_avgprec)
        print("--- Average valid roc auc score:         %.4f" % avg_valid_rocauc)
        print("--- Average valid min semantic distance: %.4f" % avg_valid_sdmin)
        print("--- Average valid max F-score:           %.4f" % avg_valid_fmax)
        
        logger.add_scalar('valid_loss_epoch', avg_valid_loss, epoch + 1)
        logger.add_scalar('valid_avgprec_epoch', avg_valid_avgprec, epoch + 1)
        logger.add_scalar('valid_rocauc_epoch', avg_valid_rocauc, epoch + 1)
        logger.add_scalar('valid_sdmin_epoch', avg_valid_sdmin, epoch + 1)
        logger.add_scalar('valid_fmax_epoch', avg_valid_fmax, epoch + 1)
        
        # LR scheduler on plateau (based on validation loss)
        if lr_sched:
            scheduler.step(avg_valid_loss)

    print("[*] Finish training.")


def evaluate(device, net, criterion, eval_loader, icvec, nth=10):
    # Eval each sample
    net.eval()
    avg_loss = 0.0
    y_true = []
    y_pred_sigm = []
    with torch.no_grad():   # set all 'requires_grad' to False
        for data in eval_loader:
            # Get current batch and transfer to device
            data = data.to(device)
            labels = data.y

            # Forward pass
            _, outputs = net(data)
            current_loss = criterion(outputs, labels)
            avg_loss += current_loss.item() / len(eval_loader)
            y_true.append(labels.cpu().numpy().squeeze())
            y_pred_sigm.append(torch.sigmoid(outputs).cpu().numpy().squeeze())

        # Calculate evaluation metrics
        y_true = np.vstack(y_true)
        y_pred_sigm = np.vstack(y_pred_sigm)

        # Average precision score
        avg_avgprec = average_precision_score(y_true, y_pred_sigm, average='samples')
        # ROC AUC score
        ii = np.where(np.sum(y_true, 0) > 0)[0]
        avg_rocauc = roc_auc_score(y_true[:, ii], y_pred_sigm[:, ii], average='macro')
        # Minimum semantic distance
        avg_sdmin = smin(y_true, y_pred_sigm, icvec, nrThresholds=nth)
        # Maximum F-score
        avg_fmax = fmax(y_true, y_pred_sigm, nrThresholds=nth)

    return avg_loss, avg_avgprec, avg_rocauc, avg_sdmin, avg_fmax, y_true, y_pred_sigm


def test(device, net, criterion, model_file, test_loader, icvec, save_file=None):
    # Load pretrained model
    epoch_num = load_checkpoint(net, filename=model_file)
    
    # Evaluate model
    avg_test_loss, avg_test_avgprec, avg_test_rocauc, avg_test_sdmin, avg_test_fmax, y_true, y_pred_sigm = evaluate(device, net, criterion, test_loader, icvec, nth=51)

    # Save predictions
    if save_file is not None:
        pickle.dump({'y_true': y_true, 'y_pred': y_pred_sigm}, open(save_file, 'wb'))

    # Display evaluation metrics
    print("--- Average test loss:                  %.4f" % avg_test_loss)
    print("--- Average test avg precision score:   %.4f" % avg_test_avgprec)
    print("--- Average test roc auc score:         %.4f" % avg_test_rocauc)
    print("--- Average test min semantic distance: %.4f" % avg_test_sdmin)
    print("--- Average test max F-score:           %.4f" % avg_test_fmax)



def extract(device, net, model_file, names_file, loader, save_file=None):
    # Load pretrained model
    epoch_num = load_checkpoint(net, filename=model_file)

    # Load names file
    names = np.loadtxt(names_file, dtype='str')

    # Extract embeddings
    net.eval()
    embeddings = {}
    with torch.no_grad():   # set all 'requires_grad' to False
        for i, data in enumerate(loader):
            # Get current batch and transfer to device
            data = data.to(device)

            # Forward pass
            emb, _ = net(data)
            embeddings[names[i]] = emb.cpu().numpy().squeeze()

        # Save file
        with open(save_file, 'wb') as f:
            pickle.dump(embeddings, f)


def load_checkpoint(net, optimizer=None, scheduler=None, filename='model_last.pth.tar'):
    start_epoch = 0
    try:
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("\n[*] Loaded checkpoint at epoch %d" % start_epoch)
    except:
        print("[!] No checkpoint found, start epoch 0")

    return start_epoch
