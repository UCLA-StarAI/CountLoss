import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from count_loss_funcs.dp_loss_funcs import *
from utils import progress_bar


def ramp_loss(out, y, device): 
    loss = torch.max(torch.min(1 - torch.mul(2*y -1, out),\
        other= torch.tensor([2],dtype=torch.float).to(device)),\
        other=torch.tensor([0],dtype=torch.float).to(device)).mean()
    return loss

def sigmoid_loss(out, y): 
    # loss = torch.gather(out, dim=1, index=y).sum()
    loss = out.gather(1, 1- y.unsqueeze(1)).mean()
    return loss


def train_PN(epoch, net, u_trainloader, optimizer, criterion, device, show_bar=True):

    if show_bar:     
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, ( _, inputs, _, targets ) in enumerate(u_trainloader):
        optimizer.zero_grad()
        
        inputs , targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)
        
        if show_bar: 
            progress_bar(batch_idx, len(u_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100.*correct/total


def validate(epoch, net, u_validloader, criterion, device, threshold, logistic=True, show_bar=True, separate=False):
    
    if show_bar:     
        print('\nTest Epoch: %d' % epoch)
    
    net.eval() 
    test_loss = 0
    correct = 0
    total = 0

    pos_correct = 0
    neg_correct = 0

    pos_total = 0
    neg_total = 0

    if not logistic: 
        # print("here")
        criterion = sigmoid_loss

    with torch.no_grad():
        all_u_outputs = []
        len_u_outputs = []
        for batch_idx, (_, inputs, _, true_targets) in enumerate(u_validloader):
            
            inputs , true_targets = inputs.to(device), true_targets.to(device)
            outputs = net(inputs)
            

            predicted  = torch.nn.functional.softmax(outputs, dim=-1)[:,0] \
                    <= torch.tensor([threshold]).to(device)

            if not logistic: 
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                
            loss = criterion(outputs, true_targets)
            

            test_loss += loss.item()
            total += true_targets.size(0)
            
            all_u_outputs.append(torch.unsqueeze(torch.nn.functional.softmax(outputs, dim=-1)[:,0].log(), 0))
            len_u_outputs.append(predicted.shape[0])
            correct_preds = predicted.eq(true_targets).cpu().numpy()
            correct += np.sum(correct_preds)

            if separate: 

                true_numpy = true_targets.cpu().numpy().squeeze()
                pos_idx = np.where(true_numpy==0)[0]
                neg_idx = np.where(true_numpy==1)[0]

                pos_correct += np.sum(correct_preds[pos_idx])
                neg_correct += np.sum(correct_preds[neg_idx])

                pos_total += len(pos_idx)
                neg_total += len(neg_idx)

            if show_bar: 
                progress_bar(batch_idx, len(u_validloader) , 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    if not separate: 
        return 100.*correct/total
    else: 
        return 100.*correct/total, 100.*pos_correct/pos_total, 100.*neg_correct/neg_total



def train(epoch, net, p_trainloader, u_trainloader, optimizer, criterion, device, show_bar=True):
    
    if show_bar:     
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        optimizer.zero_grad()
        _, p_inputs, p_targets = p_data
        _, u_inputs, u_targets, u_true_targets = u_data

        p_targets = p_targets.to(device)
        u_targets = u_targets.to(device)

       
        inputs =  torch.cat((p_inputs, u_inputs), dim=0)
        targets =  torch.cat((p_targets, u_targets), dim=0)
        inputs = inputs.to(device)
        outputs = net(inputs)
        p_outputs = outputs[:len(p_targets)]
        u_outputs = outputs[len(p_targets):]
        

        p_loss = criterion(p_outputs, p_targets)
        u_loss = criterion(u_outputs, u_targets)
        loss = (p_loss + u_loss)/2.0
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)

        if show_bar: 
            progress_bar(batch_idx, len(p_trainloader) + len(u_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        

    return 100.*correct/total
    
def validate_count(epoch, net, p_validloader, u_validloader, device, p_loss_func, u_loss_func, prop, loss_weight1=1, loss_weight2=1, show_bar=True):
    if show_bar:     
        print('\nTest Epoch: %d' % epoch)
    
    net.eval() 
    # test loss
    test_loss_total = 0
    test_loss_p = 0
    test_loss_u = 0

    # total correct
    correct_total_pu = 0
    correct_p_samples = 0
    correct_u_samples = 0
    total_expected_value = 0
    correct_u_samples_true = 0

    # number of each type of sample
    total = 0
    total_p = 0
    total_u = 0

    size = 0
    with torch.no_grad():
        for batch_idx, (p_data, u_data ) in enumerate(zip(p_validloader, u_validloader)):
            _, p_inputs, p_targets = p_data
            _, u_inputs, u_targets, u_true_targets = u_data
            
            # get number of pos and unlabeled instances
            num_pos_instances = p_inputs.shape[0]
            num_u_instances = u_inputs.shape[0]

            # get the total amount of samples
            total_u += num_u_instances
            total_p += num_pos_instances
            total += (num_u_instances + num_pos_instances)
            size+=1

            # put targets to device
            p_targets = p_targets.to(device)
            u_targets = u_targets.to(device)
            u_true_targets = u_true_targets.to(device)

            # concatenate inputs to make one model pass
            inputs =  torch.cat((p_inputs, u_inputs), dim=0)
            targets =  torch.cat((p_targets, u_targets), dim=0)

            # put input to device
            inputs = inputs.to(device)
            
            # pass to model
            outputs = net(inputs)

            # seperate out positive and unlabeled outputs
            p_outputs = outputs[:num_pos_instances]
            u_outputs = outputs[num_pos_instances:]

            # do some squeezing
            p_outputs_squeezed = torch.squeeze(p_outputs)
            u_outputs = torch.swapaxes(u_outputs, 0, 1)
            u_outputs_squeezed = torch.squeeze(u_outputs)        

            # compute losses
            p_loss = p_loss_func(p_outputs_squeezed, num_pos_instances, 1)
            u_loss = u_loss_func(u_outputs, prop, num_u_instances, device)
            loss = p_loss*loss_weight1 + u_loss*loss_weight2
            test_loss_total += loss.item()
            test_loss_p += p_loss
            test_loss_u += u_loss

            # make predictions
            total_predicted = torch.squeeze(torch.exp(outputs)).round()
            p_predicted = torch.exp(p_outputs_squeezed).round()
            u_predicted = torch.exp(u_outputs_squeezed).round()
            expected_value = torch.squeeze(torch.sum(torch.exp(u_outputs_squeezed)))
            total_expected_value += expected_value

            # eval predictions
            correct_preds = total_predicted.eq(targets).cpu().numpy()
            correct_preds_u_targets = u_predicted.eq(u_targets).cpu().numpy()
            correct_preds_u_true_targets = u_predicted.eq(u_true_targets).cpu().numpy()
            correct_preds_p_targets = p_predicted.eq(p_targets).cpu().numpy()

            correct_total_pu += np.sum(correct_preds)
            correct_p_samples += np.sum(correct_preds_p_targets)
            correct_u_samples += np.sum(correct_preds_u_targets)
            correct_u_samples_true += np.sum(correct_preds_u_true_targets)

            size = size + 1

            if show_bar: 
                progress_bar(batch_idx, len(u_validloader) , 'Loss: %.3f | Acc Rounded: %.3f%% (%d/%d)'
                        % (test_loss_total/(batch_idx+1), 100.*correct_total_pu/total, correct_total_pu, total))

    rounded_correct_pu_total = 100.*correct_total_pu/total
    rounded_correct_u_total = 100.*correct_u_samples/total_u
    rounded_correct_p_total = 100.*correct_p_samples/total_p
    rounded_correct_u_true_total = 100.*correct_u_samples_true/total_u
    tot_loss_recorded = float(test_loss_total/size)
    tot_u_loss_recorded = float(test_loss_u/size)
    tot_p_loss_recorded = float(test_loss_p/size)
    tot_loss_unweighted = tot_p_loss_recorded + tot_u_loss_recorded

    return rounded_correct_pu_total, rounded_correct_u_total, rounded_correct_u_true_total, rounded_correct_p_total, tot_loss_recorded, tot_loss_unweighted, tot_u_loss_recorded, tot_p_loss_recorded

def train_count(epoch, net, p_trainloader, u_trainloader, optimizer, p_loss_func, u_loss_func, device,  prop, loss_weight1, loss_weight2, show_bar=True):
    """"
    Train Count Loss with expected-value of binomial instead of penalizing entire count distribution
    """
    if show_bar:     
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    # train loss
    train_loss = 0

    # total correct
    correct_total_pu = 0
    correct_p_samples = 0
    correct_u_samples = 0
    total_expected_value = 0

    # number of each type of sample
    total = 0
    total_p = 0
    total_u = 0

    size = 0
    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        optimizer.zero_grad()
        _, p_inputs, p_targets = p_data
        _, u_inputs, u_targets, _ = u_data
        
        # get number of pos and unlabeled instances
        num_pos_instances = p_inputs.shape[0]
        num_u_instances = u_inputs.shape[0]

        # get the total amount of samples
        total_u += num_u_instances
        total_p += num_pos_instances
        total += (num_u_instances + num_pos_instances)
        size+=1

        # put targets to device
        p_targets = p_targets.to(device)
        u_targets = u_targets.to(device)

        # concatenate inputs to make one model pass
        inputs =  torch.cat((p_inputs, u_inputs), dim=0)
        targets =  torch.cat((p_targets, u_targets), dim=0)

        # put input to device
        inputs = inputs.to(device)
        
        # pass to model
        outputs = net(inputs)

        # seperate out positive and unlabeled outputs
        p_outputs = outputs[:num_pos_instances]
        u_outputs = outputs[num_pos_instances:]

        # do some squeezing
        p_outputs_squeezed = torch.squeeze(p_outputs)
        u_outputs = torch.swapaxes(u_outputs, 0, 1)
        u_outputs_squeezed = torch.squeeze(u_outputs)        

        # compute losses
        p_loss = 0
        p_loss = p_loss_func(p_outputs_squeezed, num_pos_instances, 1)
        u_loss = 0
        u_loss = u_loss_func(u_outputs, prop, num_u_instances, device)
        loss = p_loss*loss_weight1 + u_loss*loss_weight2
        train_loss += loss.item()

        # backpropogate
        loss.backward()
        optimizer.step()

        # make predictions
        total_predicted = torch.squeeze(torch.exp(outputs)).round()
        p_predicted = torch.exp(p_outputs_squeezed).round()
        u_predicted = torch.exp(u_outputs_squeezed).round()
        expected_value = torch.squeeze(torch.sum(torch.exp(u_outputs_squeezed)))
        total_expected_value += expected_value

        # eval predictions
        correct_preds = total_predicted.eq(targets).cpu().numpy()
        correct_preds_u_targets = u_predicted.eq(u_targets).cpu().numpy()
        correct_preds_p_targets = p_predicted.eq(p_targets).cpu().numpy()

        correct_total_pu += np.sum(correct_preds)
        correct_p_samples += np.sum(correct_preds_p_targets)
        correct_u_samples += np.sum(correct_preds_u_targets)
        
        # progress bar
        if show_bar: 
            progress_bar(batch_idx, len(p_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Expected Value: %.3f (%.3f/%.3f)'
                % (train_loss/(batch_idx+1), 100.*correct_total_pu/total, correct_total_pu, total, total_expected_value/total_u, total_expected_value, total_u))
        

    return 100.*correct_total_pu/total, 100.*correct_p_samples/total_p, 100.*correct_u_samples/total_u, train_loss/size

def validate_transformed(epoch, net, u_validloader, criterion, device, alpha, beta, show_bar=True):

    if show_bar:
        print('\nTest Epoch: %d' % epoch)
    net.eval() 
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (_, inputs, _, true_targets) in enumerate(u_validloader):
            inputs , true_targets = inputs.to(device), true_targets.to(device)
            outputs = net(inputs)
            probs  = torch.nn.functional.softmax(outputs, dim=-1)[:,0] 
            scaled_probs = alpha* (1- beta)/ beta * probs / (1-probs)
            predicted = scaled_probs <= torch.tensor([0.5]).to(device)

            loss = criterion(outputs, true_targets)
           
            test_loss += loss.item()
            total += true_targets.size(0)
            
            correct_preds = predicted.eq(true_targets).cpu().numpy()


            correct += np.sum(correct_preds)

            if show_bar: 
                progress_bar(batch_idx, len(u_validloader) , 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100.*correct/total

def train_PU_discard(epoch, net,  p_trainloader, u_trainloader, optimizer, criterion, device, keep_sample=None, show_bar=True):
    
    if show_bar:     
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        
        optimizer.zero_grad()
        
        _, p_inputs, p_targets = p_data
        u_index, u_inputs, u_targets, u_true_targets = u_data

        u_idx = np.where(keep_sample[u_index.numpy()]==1)[0]

        if len(u_idx) <1: 
            continue

        u_targets = u_targets[u_idx]

        p_targets = p_targets.to(device)
        u_targets = u_targets.to(device)
        

        u_inputs = u_inputs[u_idx]        
        inputs =  torch.cat((p_inputs, u_inputs), dim=0)
        targets =  torch.cat((p_targets, u_targets), dim=0)
        inputs = inputs.to(device)

        outputs = net(inputs)

        p_outputs = outputs[:len(p_targets)]
        u_outputs = outputs[len(p_targets):]
        
        p_loss = criterion(p_outputs, p_targets)
        u_loss = criterion(u_outputs, u_targets)

        loss = (p_loss + u_loss)/2.0

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)

        if show_bar:
            progress_bar(batch_idx, len(p_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100.*correct/total

def rank_inputs(_, net, u_trainloader, device, alpha, u_size):

    net.eval() 
    output_probs = np.zeros(u_size)
    keep_samples = np.ones_like(output_probs)
    true_targets_all = np.zeros(u_size)

    with torch.no_grad():
        for batch_num, (idx, inputs, _, true_targets) in enumerate(u_trainloader):
            idx = idx.numpy()
            
            inputs = inputs.to(device)
            outputs = net(inputs)


            probs  = torch.nn.functional.softmax(outputs, dim=-1)[:,0]         
            output_probs[idx] = probs.detach().cpu().numpy().squeeze()
            true_targets_all[idx] = true_targets.numpy().squeeze()

    sorted_idx = np.argsort(output_probs)

    keep_samples[sorted_idx[u_size - int(alpha*u_size):]] = 0

    neg_reject = np.sum(true_targets_all[sorted_idx[u_size - int(alpha*u_size):]]==1.0)

    neg_reject = neg_reject/ int(alpha*u_size)
    return keep_samples, neg_reject

def train_PU_unbiased(epoch, net,  p_trainloader, u_trainloader, optimizer, criterion, device, alpha,logistic=True, show_bar=True):
    
    if show_bar:
        print('\nTrain Epoch: %d' % epoch)
        
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if not logistic: 
        criterion = sigmoid_loss

    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        optimizer.zero_grad()

        _, p_inputs, p_targets = p_data
        _, u_inputs, u_targets, u_true_targets = u_data

        p_targets_sub = torch.ones_like(p_targets)
        p_targets, p_targets_sub, u_targets = p_targets.to(device), p_targets_sub.to(device), u_targets.to(device)


        p_inputs , u_inputs = p_inputs.to(device), u_inputs.to(device)
        targets =  torch.cat((p_targets, u_targets), dim=0)
        inputs = torch.cat((p_inputs, u_inputs), axis=0)
        outputs = net(inputs)
    
        p_outputs = outputs[:len(p_targets)]
        u_outputs = outputs[len(p_targets):]

        if not logistic: 
            p_outputs = torch.nn.functional.softmax(p_outputs, dim=-1) 
            u_outputs = torch.nn.functional.softmax(u_outputs, dim=-1)

        loss_pos = criterion(p_outputs, p_targets)
        loss_pos_neg = criterion(p_outputs, p_targets_sub)
        loss_unl = criterion(u_outputs, u_targets)

        loss = alpha * (loss_pos - loss_pos_neg) + loss_unl

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)
        
        if show_bar: 
            progress_bar(batch_idx, len(p_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return 100.*correct/total


def train_PU_nn_unbiased(epoch, net,  p_trainloader, u_trainloader, optimizer, criterion, device, alpha, logistic=True, show_bar=True):
    
    if show_bar:
        print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if not logistic: 
        criterion = sigmoid_loss

    for batch_idx, ( p_data, u_data ) in enumerate(zip(p_trainloader, u_trainloader)):
        optimizer.zero_grad()
        _, p_inputs, p_targets = p_data
        _, u_inputs, u_targets, u_true_targets = u_data

        p_targets_sub = torch.ones_like(p_targets)
        p_targets, p_targets_sub, u_targets = p_targets.to(device), p_targets_sub.to(device), u_targets.to(device)


        p_inputs , u_inputs = p_inputs.to(device), u_inputs.to(device)

        targets =  torch.cat((p_targets, u_targets), dim=0)
        inputs = torch.cat((p_inputs, u_inputs), axis=0)
        outputs = net(inputs)
        
        p_outputs = outputs[:len(p_targets)]
        u_outputs = outputs[len(p_targets):]

        if not logistic: 
            p_outputs = torch.nn.functional.softmax(p_outputs, dim=-1) 
            u_outputs = torch.nn.functional.softmax(u_outputs, dim=-1)

            # print(p_outputs)

        loss_pos = criterion(p_outputs, p_targets)
        loss_pos_neg = criterion(p_outputs, p_targets_sub)
        loss_unl = criterion(u_outputs, u_targets)


        if torch.gt((loss_unl - alpha* loss_pos_neg ), 0):
            loss = alpha * (loss_pos - loss_pos_neg) + loss_unl
        else: 
            loss = alpha * loss_pos_neg - loss_unl
        
        loss.backward()

        optimizer.step()
        # loss = alpha * (loss_pos - loss_pos_neg) + loss_unl
        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_preds = predicted.eq(targets).cpu().numpy()
        correct += np.sum(correct_preds)
        
        if show_bar:
            progress_bar(batch_idx, len(p_trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    return 100.*correct/total
