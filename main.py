import pandas as pd
from argparse import Namespace
from Vocabulary import Vocabulary
import time
from ReviewDataset import ReviewDataset
from ReviewClassifier import ReviewClassifier
dataset = ReviewDataset.load_dataset_and_make_vectorizer('reviews_with_splits_lite.csv')
dataset.save_vectorizer('vectorizer.json')
vectorizer = dataset.get_vectorizer()
classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab)) # feature 同 one-hot encoding 長度

# training
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from core import generate_batches,compute_accuracy

loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(),lr=3e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                mode='min', factor=0.5,
                                                patience=1)

DEVICE = 'cuda'
device = torch.device(DEVICE)
classifier = classifier.to(device)
try:
    for epoch in range(1000):
        dataset.set_split('train')
        batch_generator = generate_batches(dataset, 
                                           batch_size=256, 
                                           device=device)
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()
        for batch_index, batch_dict in enumerate(batch_generator):
            # step 1 梯度歸零
            optimizer.zero_grad()

            # step 2 計算輸出
            y_pred = classifier(x_in=batch_dict['x_data'].float())

            # step 3 計算loss
            loss = loss_func(y_pred,batch_dict['y_target'].float())
            loss_t = loss.item()
            running_loss = running_loss + (loss_t - running_loss) / (batch_index + 1)

            # step4 使用loss產生梯度
            loss.backward()

            # step 5 更新權重
            optimizer.step()

            #
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)
        print("train loss:%f ,train acc:%f"%(running_loss,running_acc))

except KeyboardInterrupt:
    print("exit")