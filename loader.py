import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import csv

class APTOS(Dataset):
    def __init__(self, root, csv_file, train=True, val=False, transform=None, noise_type=None, noise_rate=0.2,
                 random_state=0, img_size=(64, 64), nb_classes=5):
        self.root = os.path.expanduser(root)
        self.csv_file = csv_file
        self.train = train
        self.val = val  # Add a validation flag
        self.transform = transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.random_state = random_state
        self.img_size = img_size
        self.nb_classes = nb_classes

        # Set paths for train/val/test data
        if self.train:
            self.data_dir = os.path.join(self.root, 'train')
        elif self.val:
            self.data_dir = os.path.join(self.root, 'val')  # Add a validation directory
        else:
            self.data_dir = os.path.join(self.root, 'test')

        # Load dataset
        self.images, self.labels, self.image_names = self._load_data()

        # Store original labels for noise injection
        self.original_labels = self.labels.copy()

        # Inject label noise if specified (only for training set)
        if self.train and self.noise_type is not None and self.noise_rate > 0:
            self.labels, self.actual_noise_rate = self.noisify(
                train_labels=np.array(self.labels),
                noise_type=self.noise_type,
                noise_rate=self.noise_rate,
                random_state=self.random_state,
                nb_classes=self.nb_classes  
            )
            print("Noise type:", self.noise_type)
            print("Noise added. Actual noise rate: %.2f" % self.actual_noise_rate)
            
            self.noise_or_not = np.array(self.labels) == np.array(self.original_labels)
        else:
            self.noise_or_not = np.ones(len(self.labels), dtype=bool)

    def _load_data(self):
        """
        Loads image paths, labels, and image names from the dataset directory using a CSV file.
        """
        images = []
        labels = []
        image_names = []

        # Read CSV file
        df = pd.read_csv(self.csv_file)

        for _, row in df.iterrows():
            img_name = row['id_code']
            label = row['diagnosis']  

            # Append file extension if missing
            if not img_name.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                img_name += '.png'  # or '.jpg', depending on your file format

            img_path = os.path.join(self.data_dir, img_name)
            if os.path.exists(img_path):
                images.append(img_path)
                labels.append(label)
                image_names.append(img_name)
            else:
                print(f"Warning: Image {img_name} not found in {self.data_dir}")

        return images, labels, image_names

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the item to fetch.
        Returns:
            tuple: (image, target, index)
        """
        img_path, target, original_label, img_name = (
            self.images[index],
            self.labels[index],  # Use the noisy labels stored during initialization
            self.original_labels[index],
            self.image_names[index]
        )

        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        #img = img.resize(self.img_size)  # resize if necessary

        # Apply transforms if specified
        if self.transform is not None:
            img = self.transform(img)

        #return img, target, index, original_label  #for coteaching
        return img, target      #modified for cleanlab
        #return img, target, index   #for cbs

    def __len__(self):
        return len(self.images)

   
    def save_labels_to_file(self, original_labels, noisy_labels, file_path):
        """
        Save original and noisy labels to a CSV file.
        Args:
            original_labels (np.array): Array of original labels.
            noisy_labels (np.array): Array of noisy labels.
            file_path (str): Path to the output CSV file.
        """
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Image_Name", "Original_Label", "Noisy_Label"])
            
            for img_name, original_label, noisy_label in zip(self.image_names, original_labels, noisy_labels):
                writer.writerow([img_name, original_label, noisy_label])

    def noisify(self, train_labels, noise_type, noise_rate, random_state=0, nb_classes=None):
        """
        Inject label noise into the dataset.
        Args:
            train_labels (np.array): Array of true labels.
            noise_type (str): Type of noise to inject ('symmetric' or 'pairflip').
            noise_rate (float): Proportion of labels to corrupt.
            random_state (int): Random seed for reproducibility.
            nb_classes (int): Number of classes in the dataset.
        Returns:
            noisy_labels (np.array): Array of noisy labels.
            actual_noise_rate (float): Actual noise rate after injection.
        """
        if nb_classes is None:
            nb_classes = self.nb_classes  # Use the instance attribute if nb_classes is not provided

        if noise_type == 'pairflip':
            noisy_labels, actual_noise_rate = self.noisify_pairflip(train_labels, noise_rate, random_state, nb_classes)
        elif noise_type == 'symmetric':
            noisy_labels, actual_noise_rate = self.noisify_multiclass_symmetric(train_labels, noise_rate, random_state, nb_classes)
        elif noise_type == 'structured':
            noisy_labels, actual_noise_rate = self.noisify_structured(train_labels, noise_rate, random_state, nb_classes)
        else:
            print("no noise instructions given, using clean labels")
            noisy_labels, actual_noise_rate = train_labels, 0.0

        # Save original and noisy labels to a file
        self.save_labels_to_file(self.original_labels, noisy_labels, "noisy_labels.csv")

        return noisy_labels, actual_noise_rate

    def noisify_pairflip(self, y_train, noise, random_state=None, nb_classes=None):
        """Inject pairflip noise into labels."""
        P = np.eye(nb_classes)
        n = noise
        print("nb clases", nb_classes)
        if n > 0.0:
            # 0 -> 1
            P[0, 0], P[0, 1] = 1. - n, n
            for i in range(1, nb_classes - 1):
                P[i, i], P[i, i + 1] = 1. - n, n
            P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

            y_train_noisy = self.multiclass_noisify(y_train, P=P, random_state=random_state)
            actual_noise = (y_train_noisy != y_train).mean()
            print('Actual noise %.2f' % actual_noise)
            y_train = y_train_noisy
        
        print("Pairflip Transition Matrix:\n", P)

        return y_train, actual_noise

    def noisify_multiclass_symmetric(self, y_train, noise, random_state=None, nb_classes=None):
        """Inject symmetric noise into labels."""
        P = np.ones((nb_classes, nb_classes))
        n = noise
        P = (n / (nb_classes - 1)) * P
        print("nbclasses", nb_classes)
        if n > 0.0:
            P[0, 0] = 1. - n
            for i in range(1, nb_classes - 1):
                P[i, i] = 1. - n
            P[nb_classes - 1, nb_classes - 1] = 1. - n

            y_train_noisy = self.multiclass_noisify(y_train, P=P, random_state=random_state)
            actual_noise = (y_train_noisy != y_train).mean()
            print('Actual noise %.2f' % actual_noise)
            y_train = y_train_noisy
        
        print(" Transition Matrix:\n", P)

        return y_train, actual_noise

    def multiclass_noisify(self, y, P, random_state=0):
        """Flip classes according to transition probability matrix P."""
        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

    def noisify_structured(self, y_train, noise, random_state=None, nb_classes=None):
        P = np.eye(nb_classes) * (1 - noise)  # Identity matrix scaled for self-label probability
        n_adj = noise * 0.8  # Higher probability for adjacent classes
        n_other = noise * 0.2 / (nb_classes - 2) if nb_classes > 2 else 0  # Lower probability for farther classes

        for i in range(nb_classes):
            if i > 0:
                P[i, i - 1] = n_adj / 2  # Left adjacent class
            if i < nb_classes - 1:
                P[i, i + 1] = n_adj / 2  # Right adjacent class
            for j in range(nb_classes):
                if j != i and abs(i - j) > 1:
                    P[i, j] = n_other  # Low probability for distant classes

        P /= P.sum(axis=1, keepdims=True)  # Normalize probabilities
        y_train_noisy = self.multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        print('Actual noise %.2f' % actual_noise)
        print("Structured Transition Matrix:\n", P)
        return y_train_noisy, actual_noise
