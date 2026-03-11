import os
import json
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import math
import random

# --- 1. CONFIGURATION ---
# Define hyperparameters and file paths in a single, clean dictionary
# This makes the code easier to configure and manage.
CONFIG = {
    # Data Paths
    'base_data_path': '/content/drive/MyDrive/Google_AI_Studio/ARCAGI2025/data',
    'input_directory': 'GridTransitionDataset/training_transformed_unique_ids',

    # Model Hyperparameters
    'vocab_size': 12,  # 0-9 for colors, plus start and end tokens
    'max_seq_len': 1802, # Max length for input/output sequence
    'd_model': 512,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1,

    # Training Parameters
    'batch_size': 32,
    'learning_rate_g': 0.0002,
    'learning_rate_d': 0.0002,
    'num_epochs': 5, # Epochs per fold
    'n_splits': 5, # K-Fold splits
    'gamma_d': 0.5, # Learning rate decay for discriminator
    'gamma_g': 0.5, # Learning rate decay for generator
    'train_generator_every': 2, # Train G every N steps
}

# --- 2. DATASET CLASS ---
# A memory-efficient PyTorch Dataset that loads files on demand.
class ARCDataset(Dataset):
    def __init__(self, data_dir, file_paths):
        self.data_dir = data_dir
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        full_path = os.path.join(self.data_dir, file_path)

        with open(full_path, 'r') as f:
            task = json.load(f)

        # Assuming each task has at least one training example
        example = random.choice(task['train'])
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])

        # Flatten and concatenate the grids
        # Add special tokens for start and separation
        start_token, sep_token = 10, 11
        input_flat = input_grid.flatten()
        output_flat = output_grid.flatten()

        sequence = np.concatenate([
            [start_token],
            input_flat,
            [sep_token],
            output_flat
        ])
        
        # Pad sequence to max length
        padded_sequence = np.pad(sequence, (0, CONFIG['max_seq_len'] - len(sequence)), 'constant', constant_values=0)
        
        return torch.tensor(padded_sequence, dtype=torch.long)

# --- 3. MODEL ARCHITECTURE ---
# Transformer-based models for Generator and Discriminator
class Generator(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.token_embedding(src)
        output = self.transformer_encoder(src)
        # Apply Gumbel-Softmax for discrete token generation
        logits = self.output_layer(output)
        return logits

class Discriminator(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.token_embedding(src)
        output = self.transformer_encoder(src)
        # Flatten the output and pass to a final linear layer
        output = torch.mean(output, dim=1) # Global average pooling
        return self.output_layer(output)

# --- 4. WEIGHTS INITIALIZATION ---
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.TransformerEncoderLayer):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

# --- 5. TRAINER CLASS ---
class GANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # This memory-efficient generator replaces the problematic file list
        file_path_iterator = glob.iglob(
            os.path.join(config['base_data_path'], config['input_directory'], '**', '*.json'),
            recursive=True
        )
        self.all_file_paths = [os.path.basename(p) for p in file_path_iterator]
        print(f"Total files found: {len(self.all_file_paths)}")

    def run_cross_validation(self):
        k_fold = KFold(n_splits=self.config['n_splits'], shuffle=True, random_state=42)
        
        fold_discriminator_losses = []
        fold_generator_losses = []
        
        for fold, (train_ids, val_ids) in enumerate(k_fold.split(self.all_file_paths)):
            print(f"--- FOLD {fold+1}/{self.config['n_splits']} ---")
            
            # Create fresh models and optimizers for each fold
            generator = Generator(
                self.config['vocab_size'], self.config['d_model'], self.config['nhead'],
                self.config['num_layers'], self.config['dim_feedforward'], self.config['dropout']
            ).to(self.device)
            discriminator = Discriminator(
                self.config['vocab_size'], self.config['d_model'], self.config['nhead'],
                self.config['num_layers'], self.config['dim_feedforward'], self.config['dropout']
            ).to(self.device)

            generator.apply(weights_init)
            discriminator.apply(weights_init)

            optimizer_G = optim.Adam(generator.parameters(), lr=self.config['learning_rate_g'])
            optimizer_D = optim.Adam(discriminator.parameters(), lr=self.config['learning_rate_d'])
            criterion = nn.BCEWithLogitsLoss()
            
            # Split data into train and validation sets for this fold
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            
            dataset = ARCDataset(os.path.join(self.config['base_data_path'], self.config['input_directory']), self.all_file_paths)
            
            train_loader = DataLoader(dataset, batch_size=self.config['batch_size'], sampler=train_subsampler)
            val_loader = DataLoader(dataset, batch_size=self.config['batch_size'], sampler=val_subsampler)
            
            # Run the training process for the current fold
            avg_fold_disc_loss, avg_fold_gen_loss = self.train_gan(generator, discriminator, optimizer_G, optimizer_D, criterion, train_loader)
            
            fold_discriminator_losses.append(avg_fold_disc_loss)
            fold_generator_losses.append(avg_fold_gen_loss)

        # Print final cross-validation results
        print("\n--- Cross-Validation Results ---")
        avg_disc_loss_cv = sum(fold_discriminator_losses) / self.config['n_splits']
        avg_gen_loss_cv = sum(fold_generator_losses) / self.config['n_splits']
        print(f"Average Discriminator Loss across {self.config['n_splits']} folds: {avg_disc_loss_cv:.4f}")
        print(f"Average Generator Loss across {self.config['n_splits']} folds: {avg_gen_loss_cv:.4f}")
            
    def train_gan(self, generator, discriminator, optimizer_G, optimizer_D, criterion, data_loader):
        # A simple, balanced GAN training loop
        
        fold_disc_losses_epoch = []
        fold_gen_losses_epoch = []
        
        for epoch in range(self.config['num_epochs']):
            for i, real_sequences in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")):
                real_sequences = real_sequences.to(self.device)
                batch_size = real_sequences.size(0)

                # --- Train Discriminator ---
                discriminator.zero_grad()
                
                # Train with real data
                real_labels = torch.ones(batch_size, 1, device=self.device)
                d_output_real = discriminator(real_sequences)
                d_loss_real = criterion(d_output_real, real_labels)
                d_loss_real.backward()
                
                # Train with fake data
                noise = torch.randn(batch_size, self.config['max_seq_len'], self.config['d_model'], device=self.device)
                fake_sequences_logits = generator(real_sequences)
                fake_sequences = F.gumbel_softmax(fake_sequences_logits, tau=0.5, hard=True, dim=-1)
                
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                d_output_fake = discriminator(fake_sequences)
                d_loss_fake = criterion(d_output_fake, fake_labels)
                d_loss_fake.backward()

                # Update Discriminator weights
                d_loss = d_loss_real + d_loss_fake
                optimizer_D.step()

                # --- Train Generator ---
                # A common practice to train the generator more often than the discriminator
                if i % self.config['train_generator_every'] == 0:
                    generator.zero_grad()
                    gen_labels = torch.ones(batch_size, 1, device=self.device)
                    g_output = discriminator(generator(real_sequences))
                    g_loss = criterion(g_output, gen_labels)
                    g_loss.backward()
                    optimizer_G.step()
                
            fold_disc_losses_epoch.append(d_loss.item())
            fold_gen_losses_epoch.append(g_loss.item())

        avg_fold_discriminator_loss = sum(fold_disc_losses_epoch) / len(fold_disc_losses_epoch)
        avg_fold_generator_loss = sum(fold_gen_losses_epoch) / len(fold_gen_losses_epoch)

        print(f"--- Finished Fold ---")
        print(f"Average Discriminator Loss: {avg_fold_discriminator_loss:.4f}")
        print(f"Average Generator Loss: {avg_fold_gen_loss:.4f}")

        return avg_fold_discriminator_loss, avg_fold_gen_loss
