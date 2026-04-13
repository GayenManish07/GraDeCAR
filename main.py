import torch
import numpy as np
import csv
import pickle
import os
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from cleanlab.filter import find_label_issues
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights, resnet152
#from torchvision.models.vision_transformer import vit_b_16     #uncomment for using VIT
import torchvision.transforms as transforms
from aptos_loader import APTOS
from sklearn.model_selection import KFold
#import timm
from collections import Counter
import cv2
from PIL import Image
import argparse
from torchvision.models import resnet50, ResNet50_Weights
import math
import copy
import torch.nn.functional as F

#from torch.cuda.amp import autocast as autocast
#from cbs_utils import *

def get_resnet_model(pretrained=True):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model.to(device)

def parse_args():
    parser = argparse.ArgumentParser(description="Training config")

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Computation device')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--contrastive_rounds', type=int, default=10, help='Number of contrastive learning rounds')
    parser.add_argument('--confidence_threshold', type=float, default=0.9, help='Confidence threshold for cleanlab')
    parser.add_argument('--mixup_alpha', type=float, default=0.25, help='Alpha for mixup augmentation(strength of mixup)')
    parser.add_argument('--lambda_ce', type=float, default=0.5, help='Weight for cross entropy loss')
    parser.add_argument('--log_csv_path', type=str, default="confidence_logs.csv", help='Path to CSV log file')
    parser.add_argument('--k_folds', type=int, default=3, help='Number of folds for cross-validation')
    #parser.add_argument('--save_cleanlab_path', type=str, default='confident_samples.pkl', help='Path to save/load Cleanlab outputs')
    parser.add_argument('--noise_type', type=str, default='structured', choices=['symmetric', 'structured'], help='Type of label noise')
    parser.add_argument('--noise_rate', type=float, default=0.5, help='Rate of noisy labels')

    return parser.parse_args()


args = parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
num_classes = args.num_classes
batch_size = args.batch_size
n_epochs = args.n_epochs
contrastive_rounds = args.contrastive_rounds
confidence_threshold = args.confidence_threshold
mixup_alpha = args.mixup_alpha
lambda_ce = args.lambda_ce
log_csv_path = args.log_csv_path
k_folds = args.k_folds
#save_cleanlab_path = args.save_cleanlab_path
noise_type = args.noise_type
noise_rate = args.noise_rate

'''
#saves the confident sample indices
if args.save_cleanlab_path:
    save_cleanlab_path = args.save_cleanlab_path
else:
    save_cleanlab_path = f"cleanlab_outputs_{args.noise_type}_{args.noise_rate}.pkl"
'''

save_cleanlab_path = f"cleanlab_outputs_{args.noise_type}_{args.noise_rate}.pkl"  #saves the clean samples using cleanlab, change to necessary directory

# supervised contrastive loss
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-6

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask - torch.eye(batch_size).to(device)  # remove self-contrast

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask out self-contrast
        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size).to(device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        # Only positive pairs
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -mean_log_prob_pos
        return loss.mean()

# apply mixup to data
def mixup_data(x, y, alpha=mixup_alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        return x, y, y, 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# efficientnet model
def get_model(pretrained=True):
    weights = EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b5(weights=weights)
    in_features = model.classifier[1].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model.to(device)

# function to train resnet in each round
def train_resnet_epochs(cnn_model, confident_loader, cnn_optimizer, cnn_criterion, device, num_epochs):
    cnn_model.train()
    for epoch in range(num_epochs):
        total_cnn_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(confident_loader, desc=f"ResNet Training (Epoch {epoch+1})"):
            images, labels = images.to(device), labels.to(device)
            cnn_optimizer.zero_grad()
            outputs = cnn_model(images)
            loss_cnn = cnn_criterion(outputs, labels)
            loss_cnn.backward()
            cnn_optimizer.step()
            total_cnn_loss += loss_cnn.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_cnn_loss = total_cnn_loss / len(confident_loader)
        #acc = 100 * correct / total
        print(f"ResNet Epoch {epoch+1} - Loss: {avg_cnn_loss:.4f}") #, Accuracy: {acc:.2f}%")

'''
# VIT model can be used for cleanlab but in this implementation  efficientnet has been used
def get_model_cleanlab(num_classes=5):
    # Load pretrained DINO ViT backbone
    backbone = timm.create_model('vit_small_patch16_224_dino', pretrained=True, num_classes=0)
    backbone.to(device)
    backbone.eval()  # Freeze the backbone for feature extraction

    # Freeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    # Get feature dimension of the DINO backbone output
    feature_dim = backbone.num_features

    # linear classifier head
    classifier = nn.Linear(feature_dim, num_classes)

    # Combine backbone + classifier into one model
    model = nn.Sequential(backbone, classifier)
    return model.to(device)
'''

def get_cnn_model():
    model = resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

#for efficientnet
class ContrastiveWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = encoder.classifier[1].in_features if hasattr(encoder, 'classifier') else encoder.fc.in_features
        #self.feature_dim = encoder.feature_dim  #mofification for convnext

        if hasattr(encoder, 'classifier'):
            self.encoder.classifier = nn.Identity()
        elif hasattr(encoder, 'fc'):
            self.encoder.fc = nn.Identity()
        self.linear = nn.Linear(self.feature_dim, num_classes).to(device)

    def forward(self, x):
        return nn.functional.normalize(self.encoder(x), dim=1)


# Modified dataset class to support relabelling
class RelabelableDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.relabels = {}

    def relabel(self, idx, new_label):
        self.relabels[idx] = new_label

    def __getitem__(self, idx):
        img, original_label = self.dataset[idx]
        label = self.relabels.get(idx, original_label)
        return img, label

    def __len__(self):
        return len(self.dataset)

    @property
    def labels(self):
        # if the base dataset is a subset, forward to its dataset
        if isinstance(self.dataset, Subset):
            return [self.dataset.dataset.labels[i] for i in self.dataset.indices]
        # else assume base dataset has .labels
        return self.dataset.labels
    

# relabel based on high-confidence predictions with agreement from two models
def relabel_confident_samples(model1, model2, dataset, threshold=0.9):
    model1.eval()
    model2.eval()
    relabels = {}
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=16)

    with open(log_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SampleIndex", "PredictedLabel", "Confidence"])

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(loader):
                inputs = inputs.to(device)
                outputs1 = model1.encoder(inputs)
                logits1 = nn.functional.linear(outputs1, model1.linear.weight, model1.linear.bias)
                probs1 = torch.softmax(logits1, dim=1)
                confs1, preds1 = torch.max(probs1, dim=1)

                #outputs2 = model2(inputs)
                #logits2 = model2.fc(outputs2)
                #probs2 = torch.softmax(logits2, dim=1)
                #confs2, preds2 = torch.max(probs2, dim=1)

                outputs2 = model2(inputs)      
                probs2 = torch.softmax(outputs2, dim=1)  
                confs2, preds2 = torch.max(probs2, dim=1)

                for i in range(inputs.size(0)):
                    sample_idx = batch_idx * batch_size + i
                    if preds1[i] == preds2[i]:
                        confidence = (confs1[i].item() + confs2[i].item()) / 2
                        predicted_label = preds1[i].item()
                        #_, original_label = dataset.dataset[sample_idx]
                        original_label = dataset.labels[sample_idx]
                        writer.writerow([sample_idx, predicted_label, confidence])
                        if confidence > threshold and predicted_label != original_label:
                            relabels[sample_idx] = predicted_label

    for idx, new_label in relabels.items():
        dataset.relabel(idx, new_label)
    print(f"Relabeled {len(relabels)} samples in this round.")

# training
def train_contrastive_epoch(model, dataloader, contrastive_criterion, ce_criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in tqdm(dataloader, desc=f"Contrastive Training (Epoch {epoch+1}/{epochs})"):
            images, labels = images.to(device), labels.to(device)
            mixed_images, targets_a, targets_b, lam = mixup_data(images, labels)
            optimizer.zero_grad()

            features = model.encoder(mixed_images)  # Only encoder
            logits = model.linear(features)         # Classifier head

            supcon_loss = lam * contrastive_criterion(features, targets_a) + (1 - lam) * contrastive_criterion(features, targets_b)
            ce_loss = lam * ce_criterion(logits, targets_a) + (1 - lam) * ce_criterion(logits, targets_b)

            loss = (1 - lambda_ce) * supcon_loss + lambda_ce * ce_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss / len(dataloader):.4f}")
    return total_loss / len(dataloader)

# main pipeline
def main(train_dataset, test_dataset, args):

    args = args
    
    if os.path.exists(log_csv_path):
        os.remove(log_csv_path)

    
    if not os.path.exists(save_cleanlab_path):
        print("Running Cleanlab to get confident samples with K-Fold...")       #cleanlab used here
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        all_probs = torch.zeros((len(train_dataset), num_classes))

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            print(f"Fold {fold+1}/{k_folds}")
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=16)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=16)

            model = get_model()         #this is efficientnetv5
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(3):
                model.train()
                for x, y in tqdm(train_loader, desc=f"Train Fold {fold+1} Epoch {epoch+1}"):
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                idx = 0
                for x, _ in tqdm(val_loader, desc=f"Val Fold {fold+1}"):
                    x = x.to(device)
                    out = model(x)
                    probs = torch.softmax(out, dim=1).cpu()
                    batch_size = x.size(0)
                    all_probs[val_idx[idx:idx+batch_size]] = probs
                    idx += batch_size

        true_labels = torch.tensor([label for _, label in train_dataset])
        probs_np = all_probs.numpy()
        labels_np = true_labels.numpy()
        ranked_issues = find_label_issues(labels_np, probs_np, return_indices_ranked_by='self_confidence')
        confident_idx = sorted(list(set(range(len(train_dataset))) - set(ranked_issues)))
        print(f"Confident samples: {len(confident_idx)} / {len(train_dataset)}")

        with open(save_cleanlab_path, 'wb') as f:
            pickle.dump({"confident_idx": confident_idx}, f)

        # after confident_idx is obtained
        true_labels = [label for _, label in train_dataset]
        confident_labels = [true_labels[idx] for idx in confident_idx]
        label_counts = Counter(confident_labels)

        print("\nLabel distribution of confident samples:")
        for label, count in sorted(label_counts.items()):
            print(f"Class {label}: {count} samples")

    else:
        print("Loading Cleanlab confident indices from disk...")
        with open(save_cleanlab_path, 'rb') as f:
            confident_idx = pickle.load(f)["confident_idx"]
   
    # important part below, changing them can make the code unusable

    true_labels = train_dataset.labels
    # After confident_idx is obtained
    #true_labels = [label for _, label in train_dataset]
    confident_labels = [true_labels[idx] for idx in confident_idx]
    label_counts = Counter(confident_labels)

    print("\nLabel distribution of confident samples:")
    for label, count in sorted(label_counts.items()):
        print(f"Class {label}: {count} samples")
    

    #import csv

    # Save confident samples and their labels to a CSV
    with open("confident_samples.csv", mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Index", "Label"])  # header

        for idx in confident_idx:
            _, label = train_dataset[idx]       #added two places to accomodate cbs
            writer.writerow([idx, label])



    relabelable_dataset = RelabelableDataset(train_dataset)
    confident_subset = Subset(relabelable_dataset, confident_idx)
    confident_targets = torch.tensor(confident_labels)

    #sampler = get_weighted_sampler(confident_targets.numpy())
    confident_loader = DataLoader(confident_subset, batch_size=16, num_workers=16)

    base_model = get_model(pretrained=True)

    contrastive_model = ContrastiveWrapper(base_model)
    cnn_model = get_cnn_model()

    optimizer = optim.Adam(list(contrastive_model.parameters()) + list(contrastive_model.linear.parameters()), lr=1e-4)
    contrastive_criterion = SupConLoss()
    ce_criterion = nn.CrossEntropyLoss()           # cross entropy
    
    #cnn parameters
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4)
    cnn_criterion = nn.CrossEntropyLoss()           # cross entropy 



    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16)

    n_eval_rounds = 2  # used to control how frequently to evaluate, currently unused

    initial_epochs = n_epochs
    decay_per_round = int(0.10 * initial_epochs)  # 10% decay each relabelling round
    epoch_schedule = [initial_epochs - decay_per_round * r for r in range(contrastive_rounds)]


    for round in range(contrastive_rounds):
        print(f"\n--- Contrastive Round {round+1} ---")
        current_epochs = epoch_schedule[round]
        # Train Contrastive Model
        loss = train_contrastive_epoch(contrastive_model, confident_loader, contrastive_criterion, ce_criterion, optimizer, epochs=current_epochs)
        print(f"Contrastive Model Loss: {loss:.4f}")

        train_resnet_epochs(cnn_model, confident_loader, cnn_optimizer, cnn_criterion, device, num_epochs=current_epochs)

        # Relabel samples if both models agree
        relabel_confident_samples(contrastive_model, cnn_model, relabelable_dataset, threshold=confidence_threshold)
        
        # update confident data for next round
        confident_subset = Subset(relabelable_dataset, confident_idx)
        confident_loader = DataLoader(confident_subset, batch_size=16, shuffle=True, num_workers=16)

        # Evaluation
        contrastive_model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in test_loader:        #changed here
                x = x.to(device)
                y = y.to(device)
                features = contrastive_model(x)
                logits = contrastive_model.linear(features)
                predictions = torch.argmax(logits, dim=1)
                preds.extend(predictions.cpu().numpy())
                targets.extend(y.cpu().numpy())

        f1 = f1_score(targets, preds, average='macro')
        acc = accuracy_score(targets, preds)
        kappa = cohen_kappa_score(targets, preds, weights='quadratic')
        print(f"[Final Epoch] F1 Score: {f1:.4f}, Quadratic Weighted Kappa: {kappa:.4f}, Accuracy: {acc:.4f}")
             
                #Evaluate ResNet Model
        cnn_model.eval()
        cnn_preds, cnn_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:        #changed here
                x = x.to(device)
                y = y.to(device)
                outputs = cnn_model(x)
                predictions = torch.argmax(outputs, dim=1)
                cnn_preds.extend(predictions.cpu().numpy())
                cnn_targets.extend(y.cpu().numpy())

        cnn_f1 = f1_score(cnn_targets, cnn_preds, average='macro')
        cnn_kappa = cohen_kappa_score(cnn_targets, cnn_preds, weights='quadratic') #no need in every round
        cnn_acc = accuracy_score(cnn_targets, cnn_preds)
        

        print(f"[Final Epoch] F1 Score: {cnn_f1:.4f}, Quadratic Weighted Kappa: {cnn_kappa:.4f}, Accuracy: {cnn_acc:.4f}")
        #else:
            #print(f"[Round {round+1}] F1 Score: {cnn_f1:.4f}, Accuracy: {cnn_acc:.4f}")
            #print(f"[Round {round+1}] ResNet - F1 Score: {cnn_f1:.4f}, Kappa: {cnn_kappa:.4f}, Accuracy: {cnn_acc:.4f}")

        if round == contrastive_rounds - 1:
            cnn_model.eval()
            contrastive_model.eval()

            final_preds, final_targets = [], []

            # CSV writer
            csv_filename = f"predictions_{noise_type}_rate_{noise_rate}_dual.csv"
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["SampleIndex", "PredictedLabel"])

                with torch.no_grad():
                    for idx, (x, y) in enumerate(test_loader):      #CHANGED HERE
                        x = x.to(device)
                        y = y.to(device)

                        # CNN model logits
                        cnn_logits = cnn_model(x)

                        # Contrastive model logits
                        features = contrastive_model(x)
                        contrastive_logits = contrastive_model.linear(features)

                        # Average softmax probabilities
                        cnn_probs = torch.softmax(cnn_logits, dim=1)
                        contrastive_probs = torch.softmax(contrastive_logits, dim=1)
                        avg_probs = (cnn_probs + contrastive_probs) / 2

                        # Final predictions from averaged probabilities
                        preds = torch.argmax(avg_probs, dim=1)

                        final_preds.extend(preds.cpu().numpy())
                        final_targets.extend(y.cpu().numpy())

                        # Write each sample's predictions
                        for i, p in enumerate(preds):
                            writer.writerow([idx * test_loader.batch_size + i, p.item()])

            # Evaluation
            f1 = f1_score(final_targets, final_preds, average='macro')
            acc = accuracy_score(final_targets, final_preds)
            kappa = cohen_kappa_score(final_targets, final_preds, weights='quadratic')

            print(f"[Final Round - Averaged] F1 Score: {f1:.4f}, Quadratic Weighted Kappa: {kappa:.4f}, Accuracy: {acc:.4f}")
            print(f"Predictions saved to {csv_filename}")

# main

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = APTOS(
        root='/data/UG/Manish/aptos_split', 
        csv_file= '/data/UG/Manish/aptos_split/train.csv',
        train=True, 
        noise_type=noise_type, 
        noise_rate=noise_rate,
        transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    )

    test_dataset = APTOS(
        root='/data/UG/Manish/aptos_split', 
        csv_file= '/data/UG/Manish/aptos_split/test.csv',
        train=False,
        transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    )

    main(train_dataset, test_dataset, args)
