import numpy as np
from numpy.lib.function_base import average
import pandas as pd
from sklearn import metrics
import random
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.nn.functional as F
from torch.autograd import Variable

from transformers import BertTokenizer

from torch import cuda
device = 'cuda:0' if cuda.is_available() else 'cpu'

import data_handler
from _utils import FOLD_to_TIMELINE, FOLDER_models, BERT_params



class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed.

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CustomDataset(Dataset):
    def __init__(self, posts, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.post_texts = posts
        self.targets = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.post_texts)

    def __getitem__(self, index):
        post_text = str(self.post_texts[index])
        post_text = " ".join(post_text.split())

        inputs = self.tokenizer.encode_plus(
            post_text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }
        
        
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(DOUT)
        if use_three_labels:
            self.l3 = torch.nn.Linear(768, 3)
        else:
            self.l3 = torch.nn.Linear(768, 5)
            
    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
        

def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_function(outputs, targets)
        if _%100==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

def validation(epoch):
    model.eval() 
    val_targets, val_outputs, val_loss_total = [], [], 0.0
    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            
            val_loss_total+=loss_function(outputs, targets)
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return np.array(val_outputs), np.array(val_targets), val_loss_total.item()


def apply_to_test_set(epoch):
    model.eval() 
    test_targets, test_outputs, test_loss_total = [], [], 0.0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            
            test_loss_total+=loss_function(outputs, targets)
            test_targets.extend(targets.cpu().detach().numpy().tolist())
            test_outputs.extend(outputs.cpu().detach().numpy().tolist())
    return np.array(test_outputs), np.array(test_targets), test_loss_total.item()


def convert_labels_to_categorical(labels, three_labels=True):
    '''
    Converting string labels to their categorical version.
    '''
    if three_labels:
        vals = {'0':0, 'IE':1, 'IS':2}
    else:
        vals = {'0':0, 'IE':1, 'IEP':2, 'IS':3, 'ISB':4}    
    return np.array([vals[k] for k in labels])


def convert_categorical_to_labels(categories, three_labels=True):
    '''
    Converting categorical predictions to their actual (string) class.
    '''
    if three_labels:
        vals = ['0', 'IE', 'IS']
    else:
        vals = ['0', 'IE', 'IEP', 'IS', 'ISB']
    return np.array([vals[int(k)] for k in categories])        


if __name__=='__main__':
    my_ran_seed = 12
    set_seed(my_ran_seed)
    myGenerator = torch.Generator()
    myGenerator.manual_seed(my_ran_seed)
    use_three_labels = True
    all_results = dict()

    for test_fold in range(len(FOLD_to_TIMELINE)): # for each (test) fold
        print('Training BERT (post) \tFold: '+str(test_fold+1)+'/'+str(len(FOLD_to_TIMELINE)))
        
        # Just getting the train/test data: timelines_ids, posts_id, texts, labels
        test_tl_ids, test_pids, test_texts, test_labels = data_handler.get_timelines_for_fold(test_fold)
        train_tl_ids, train_pids, train_texts, train_labels = data_handler.get_timelines_except_for_fold(test_fold)

        # Converting to a 3-label prediction task
        if use_three_labels:
            train_labels, test_labels = data_handler.get_three_labels(train_labels, test_labels)
        Ytrain = convert_labels_to_categorical(train_labels)
        Ytest = convert_labels_to_categorical(test_labels)

        # Creating the datasets to be used by CustomDataset and DataLoader:
        df = pd.DataFrame(list(zip(train_texts, Ytrain)), columns =['post', 'label'])
        train_size = 0.67
        train_dataset = df.sample(frac=train_size,random_state=my_ran_seed)
        val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)
        test_df = pd.DataFrame(list(zip(test_texts, Ytest)), columns =['post', 'label'])
        
        # param search grid
        best_eval_score = -1.0
        num_trial = 0
        best_lr = 'nan'

        for MAX_LEN in BERT_params['max_len']:
            for LEARNING_RATE in BERT_params['lr']:
                for DOUT in BERT_params['dout']:
                    
                    my_key = str(test_fold)+'_'+str(LEARNING_RATE)+'_'+str(DOUT)
                    num_trial+=1

                    TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, EPOCHS = BERT_params['train_batch'], BERT_params['valid_batch'], BERT_params['epochs']
                    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

                    train_params = {'batch_size': TRAIN_BATCH_SIZE,'shuffle': True,'num_workers': 0, 'generator':myGenerator, 'worker_init_fn':0}
                    val_params = {'batch_size': VALID_BATCH_SIZE,'shuffle': False,'num_workers': 0, 'generator':myGenerator, 'worker_init_fn':0}
                    test_params = {'batch_size': VALID_BATCH_SIZE,'shuffle': False,'num_workers': 0, 'generator':myGenerator, 'worker_init_fn':0}

                    training_set = CustomDataset(train_dataset.post.values, train_dataset.label.values, tokenizer, MAX_LEN)
                    validation_set = CustomDataset(val_dataset.post.values, val_dataset.label.values, tokenizer, MAX_LEN)
                    testing_set = CustomDataset(test_df.post.values, test_df.label.values, tokenizer, MAX_LEN)
                    
                    training_loader = DataLoader(training_set, **train_params)
                    validation_loader = DataLoader(validation_set, **val_params)
                    testing_loader = DataLoader(testing_set, **test_params)

                    #focal alpha:
                    tr_ = np.array(train_dataset.label.values)
                    class_proba = [len(np.where(tr_==0)[0])/len(tr_), len(np.where(tr_==1)[0])/len(tr_), len(np.where(tr_==2)[0])/len(tr_)]
                    inv_class_proba = np.sqrt(1/np.array(class_proba))
                    #inv_class_proba = (1/np.array(class_proba))
                    
                    alphas_focal = list(inv_class_proba)
                    print(alphas_focal)

                    # Defining the model
                    model = BERTClass()
                    model.to(device)
                    loss_function = FocalLoss(gamma=2, alpha=alphas_focal) #focal loss
                    #loss_function = FocalLoss(gamma=2) #focal loss
                    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
                    
                    #prev_loss_val = 10000

                    # Fine-tuning the model and validating
                    for epoch in range(EPOCHS):
                        train(epoch)

                        # make preds on validation set and measure loss/accuracy
                        preds, actual, current_eval_loss = validation(epoch)
                        preds = np.argmax(preds, axis=1)
                        f1_macro_dev = metrics.f1_score(actual, preds, average='macro') #adtsakal

                        print(num_trial, 'Prev Best LR/loss:', best_lr, '\t', best_eval_score, '\n',
                        'Current Eval_loss:', current_eval_loss, '\tF1-Macro (dev):', f1_macro_dev)


                        if f1_macro_dev>best_eval_score: #if best model, save
                            best_lr = str(DOUT)+'_'+str(LEARNING_RATE)

                            preds, actual, _ = apply_to_test_set(epoch)
                            preds = np.argmax(preds, axis=1)

                            ac_lbl = convert_categorical_to_labels(actual, use_three_labels)
                            pr_lbl = convert_categorical_to_labels(preds, use_three_labels)
                            data_handler.save_results([model, tokenizer], test_tl_ids, test_pids, ac_lbl, pr_lbl, 'bert', test_fold, use_three_labels, 'bertnew5_post_focalgamma2alphainversesqrt')

                            print('F1-Macro (test):', metrics.f1_score(actual, preds, average='macro'), '\t(Saved)')
                            best_eval_score = f1_macro_dev
                        #if current_eval_loss>prev_loss_val: #stop if val_loss increases
                        #    break
                        #prev_loss_val = current_eval_loss