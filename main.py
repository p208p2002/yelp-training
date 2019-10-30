import pandas as pd
from Vocabulary import Vocabulary
import time
from ReviewDataset import ReviewDataset
from ReviewClassifier import ReviewClassifier
dataset = ReviewDataset.load_dataset_and_make_vectorizer('reviews_with_splits_lite.csv')
dataset.save_vectorizer('vectorizer.json')
vectorizer = dataset.get_vectorizer()
classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab)) # feature 同 one-hot encoding 長度

args = Namespace(   
    seed=1337,    
    cuda=True,
)

# training
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from generate_batches import generate_batches

loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(),lr=3e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                mode='min', factor=0.5,
                                                patience=1)


try:
    for epoch in 1000:
        dataset.set_split('train')
        batch_generator = generate_batches(dataset, 
                                           batch_size=256, 
                                           device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()
except KeyboardInterrupt:
    print("exit")