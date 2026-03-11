import os
import json
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import math
import random

# --- 1. CONFIGURATION ---
CONFIG = {
    # Data Paths
    'base_data_path': '/content/drive/MyDrive/Google_AI_Studio/ARCAGI2025/data',
    'input_directory': 'GridTransitionDataset/training_transformed_unique_ids',

    # Model Hyperparameters
    'vocab_size': 12,
    'max_seq_len': 1802,
    'd_model': 512,
    'nhead': 8,
    'num_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1,

    # Training Parameters
    'batch_size': 32,
    'learning_rate_g': 0.0002,
    'learning_rate_d': 0.0002,
    'num_epochs': 5,
    'n_splits': 5,
    'train_generator_every': 2,
}

# --- 2. DATASET CLASS ---
class ARCDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with open(file_path, 'r') as f:
            task = json.load(f)

        # Assuming a single, deterministic example for simplicity and reproducibility
        example = task['train'][0]
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])

        # Flatten and concatenate the grids
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
        logits = self.output_layer(output)
        # Apply Gumbel-Softmax for discrete token generation in a separate step
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
        output = torch.mean(output, dim=1)
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
        
        # FIX: Collect full file paths, not just basenames
        file_path_iterator = glob.iglob(
            os.path.join(self.config['base_data_path'], self.config['input_directory'], '**', '*.json'),
            recursive=True
        )
        self.all_file_paths = list(file_path_iterator)
        print(f"Total files found: {len(self.all_file_paths)}")

    def run_cross_validation(self):
        k_fold = KFold(n_splits=self.config['n_splits'], shuffle=True, random_state=42)
        
        fold_discriminator_losses = []
        fold_generator_losses = []
        
        for fold, (train_ids, val_ids) in enumerate(k_fold.split(self.all_file_paths)):
            print(f"--- FOLD {fold+1}/{self.config['n_splits']} ---")
            
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
            
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            
            # FIX: Pass the correct file paths to the Dataset
            dataset = ARCDataset(self.all_file_paths)
            
            train_loader = DataLoader(dataset, batch_size=self.config['batch_size'], sampler=train_subsampler)
            val_loader = DataLoader(dataset, batch_size=self.config['batch_size'], sampler=val_subsampler)
            
            avg_fold_disc_loss, avg_fold_gen_loss = self.train_gan(generator, discriminator, optimizer_G, optimizer_D, criterion, train_loader)
            
            fold_discriminator_losses.append(avg_fold_disc_loss)
            fold_generator_losses.append(avg_fold_gen_loss)

        print("\n--- Cross-Validation Results ---")
        avg_disc_loss_cv = sum(fold_discriminator_losses) / self.config['n_splits']
        avg_gen_loss_cv = sum(fold_generator_losses) / self.config['n_splits']
        print(f"Average Discriminator Loss across {self.config['n_splits']} folds: {avg_disc_loss_cv:.4f}")
        print(f"Average Generator Loss across {self.config['n_splits']} folds: {avg_gen_loss_cv:.4f}")
            
    def train_gan(self, generator, discriminator, optimizer_G, optimizer_D, criterion, data_loader):
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
                
                # --- Train with fake data ---
                # FIX: Create fake data from random noise, not real data.
                noise = torch.randn(batch_size, self.config['max_seq_len'], self.config['d_model'], device=self.device)
                fake_sequences_logits = generator(noise) # Pass noise to the generator

                # Apply Gumbel-Softmax to get discrete tokens.
                # Use argmax to get the indices that the discriminator expects.
                fake_sequences_one_hot = F.gumbel_softmax(fake_sequences_logits, tau=0.5, hard=True, dim=-1)
                fake_sequences = torch.argmax(fake_sequences_one_hot, dim=-1)

                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                d_output_fake = discriminator(fake_sequences)
                d_loss_fake = criterion(d_output_fake, fake_labels)
                d_loss_fake.backward()

                # Update Discriminator weights
                d_loss = d_loss_real + d_loss_fake
                optimizer_D.step()

                # --- Train Generator ---
                if i % self.config['train_generator_every'] == 0:
                    generator.zero_grad()
                    gen_labels = torch.ones(batch_size, 1, device=self.device)
                    # FIX: Train the generator on new fake data from noise
                    fake_sequences_logits = generator(noise)
                    # The discriminator needs the output of the Gumbel-Softmax as integer indices
                    fake_sequences = torch.argmax(F.gumbel_softmax(fake_sequences_logits, tau=0.5, hard=True, dim=-1), dim=-1)
                    g_output = discriminator(fake_sequences)
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

# --- 6. MAIN EXECUTION ---
if __name__ == '__main__':
    trainer = GANTrainer(CONFIG)
    trainer.run_cross_validation()