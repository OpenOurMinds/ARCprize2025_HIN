import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import GlorotNormal
import os
import json
import numpy as np
import logging
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import logging
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import GlorotNormal

# Configure logging for better output
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if running in Google Colab and mount Drive for portability
def mount_google_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        logging.info("Google Drive mounted successfully.")
    except ImportError:
        logging.warning("Not running in Google Colab. Skipping Google Drive mount.")


class BaseDataLoader:
    """Base class for creating a tf.data.Dataset pipeline."""
    def __init__(self, config):
        self.config = config

    def _pad_grid_tf(self, grid_tensor):
        """Pads or crops a grid tensor to a fixed size."""
        height, width = tf.shape(grid_tensor)[0], tf.shape(grid_tensor)[1]

        # Calculate padding amounts (can be negative if the grid is larger)
        pad_height = self.config.grid_size - height
        pad_width = self.config.grid_size - width

        # Pad if smaller, crop if larger
        padded_grid = tf.pad(
            grid_tensor,
            [[tf.maximum(0, pad_height), tf.maximum(0, 0)],  # Pad top, no pad bottom (will crop from bottom)
             [tf.maximum(0, pad_width), tf.maximum(0, 0)]],   # Pad left, no pad right (will crop from right)
            constant_values=0
        )

        # Crop to the exact grid_size
        padded_grid = padded_grid[:self.config.grid_size, :self.config.grid_size]

        return tf.cast(padded_grid, dtype=tf.int32)


    def _get_data_generator(self):
        """Placeholder for a generator function, implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_data_generator.")

    def load_dataset(self):
        """Returns a batched, pre-processed tf.data.Dataset."""
        generator = self._get_data_generator()
        # Define output signature with a variable shape initially
        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(shape=(None, None), dtype=tf.int32)
        )

        # Filter out any potentially invalid items yielded by the generator
        # Check rank and ensure dimensions are positive before padding
        ds = ds.filter(lambda x: tf.rank(x) == 2 and tf.shape(x)[0] > 0 and tf.shape(x)[1] > 0)

        # Map padding and expanding dimensions
        ds = ds.map(self._pad_grid_tf, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x: tf.expand_dims(x, axis=-1), num_parallel_calls=tf.data.AUTOTUNE)

        # Explicitly set the shape and filter out any elements that don't match
        # This ensures all elements have the target shape before batching
        target_shape = tf.constant([self.config.grid_size, self.config.grid_size, 1], dtype=tf.int32)
        ds = ds.map(lambda x: tf.ensure_shape(x, target_shape))

        # Corrected filtering: use tf.reduce_all to get a scalar boolean
        ds = ds.filter(lambda x: tf.reduce_all(tf.equal(tf.shape(x), target_shape)))

        ds = ds.shuffle(buffer_size=5000)
        ds = ds.batch(self.config.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds



# --- D-VAE Model Definition ---
class ARCConfig:
    def __init__(self):
        self.project_root = '/content/drive/MyDrive/Google_AI_Studio/ARCAGI2025'
        self.datasets_dir = os.path.join(self.project_root, 'datasets')
        self.vae_checkpoints_dir = os.path.join(self.project_root, 'vae_checkpoints')
        self.training_challenges_path = os.path.join(self.datasets_dir, 'arc-agi_training_challenges.json')
        self.synthetic_data_paths = [
            os.path.join(self.datasets_dir, f'generated_dcgan_dataset_{size}x{size}.json')
            for size in [3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 30]
        ]
        self.grid_size = 30
        self.latent_dim = 128
        self.num_colors = 10
        self.epochs = 5
        self.batch_size = 32

class ARCDataLoader(BaseDataLoader):
    """Data loader for the official ARC challenges."""
    def _get_data_generator(self):
        def generator():
            try:
                with open(self.config.training_challenges_path, 'r') as f:
                    full_data = json.load(f)
            except FileNotFoundError:
                logging.error(f"ARC training challenges file not found at: {self.config.training_challenges_path}")
                return # Stop the generator if file not found

            for task_data in full_data.values():
                if not isinstance(task_data, dict):
                    logging.warning(f"Skipping invalid task data: {task_data}")
                    continue

                for example in task_data.get('train', []): # Use .get with default empty list
                    # Ensure example is a dictionary and has a non-empty 'output' list
                    if isinstance(example, dict) and 'output' in example and isinstance(example['output'], list) and example['output']:
                        try:
                            output_grid = np.array(example['output'], dtype=np.int32)
                            # Further validation: ensure it's a 2D array with positive dimensions
                            if output_grid.ndim == 2 and output_grid.shape[0] > 0 and output_grid.shape[1] > 0:
                                yield output_grid
                            else:
                                logging.warning(f"Skipping invalid grid shape or empty grid in ARC data: {output_grid.shape}")
                        except ValueError as e:
                            logging.warning(f"Skipping item due to ValueError during numpy conversion in ARC data: {e} - Data: {str(example['output'])[:100]}...") # Log first 100 chars of data
                        except Exception as e:
                             logging.warning(f"Skipping item due to unexpected error in ARC data generator: {e}")
                    else:
                         logging.warning(f"Skipping invalid or missing 'output' in ARC example: {example}")

        return generator


class SyntheticDataLoader(BaseDataLoader):
    """Data loader for GAN-generated synthetic grids."""
    def _get_data_generator(self):
        def generator():
            for path in self.config.synthetic_data_paths:
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logging.warning(f"Skipping file {path} due to error: {e}")
                    continue # Move to the next file

                if not isinstance(data, list):
                    logging.warning(f"Skipping file {path} because root is not a list.")
                    continue

                for item in data:
                    try:
                        # Check for 2D list format
                        if isinstance(item, list) and item and all(isinstance(sublist, list) and sublist for sublist in item):
                            grid = np.array(item, dtype=np.int32)
                            if grid.ndim == 2 and grid.shape[0] > 0 and grid.shape[1] > 0:
                                yield grid
                            else:
                                logging.warning(f"Skipping synthetic grid with invalid 2D shape: {grid.shape} in file {path}")

                        # Check for flat list format that can be reshaped
                        elif isinstance(item, list) and all(isinstance(val, (int, float)) for val in item) and len(item) > 0:
                            grid_size = int(np.sqrt(len(item)))
                            if grid_size * grid_size == len(item):
                                grid = np.array(item, dtype=np.int32).reshape(grid_size, grid_size)
                                yield grid
                            else:
                                logging.warning(f"Skipping flat list in {path} that isn't a perfect square: len={len(item)}")

                        else:
                            logging.warning(f"Skipping invalid item format in {path}: {type(item)}")

                    except ValueError as e:
                         logging.warning(f"Skipping item due to ValueError during numpy conversion/reshape in {path}: {e} - Item type: {type(item)}")
                    except Exception as e:
                         logging.warning(f"Skipping item due to unexpected error in {path}: {e} - Item type: {type(item)}")

        return generator



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(Model):
    """The encoder component of the D-VAE, refactored to use the subclassed API."""
    def __init__(self, latent_dim, grid_size, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        
        # Define layers as attributes
        self.conv1 = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same', kernel_initializer=GlorotNormal())
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer=GlorotNormal())
        self.bn2 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.z_mean_dense = layers.Dense(self.latent_dim, name='z_mean')
        self.z_log_var_dense = layers.Dense(self.latent_dim, name='z_log_var')

    def call(self, inputs):
        # Implement the forward pass in the call method
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.flatten(x)
        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)
        return z_mean, z_log_var


class Decoder(Model):
    """The decoder component of the D-VAE, refactored to use the subclassed API."""
    def __init__(self, latent_dim, num_colors, grid_size, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.num_colors = num_colors
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        
        initial_h = grid_size // 4
        initial_w = grid_size // 4
        
        # Define layers as attributes
        self.dense = layers.Dense(initial_h * initial_w * 16, activation='relu')
        self.reshape = layers.Reshape((initial_h, initial_w, 16))
        self.conv_t1 = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same', kernel_initializer=GlorotNormal())
        self.bn1 = layers.BatchNormalization()
        self.conv_t2 = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same', kernel_initializer=GlorotNormal())
        self.bn2 = layers.BatchNormalization()
        self.output_conv = layers.Conv2DTranspose(self.num_colors, 3, activation='softmax', padding='same', kernel_initializer=GlorotNormal())

    def call(self, inputs):
        # Implement the forward pass in the call method
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv_t1(x)
        x = self.bn1(x)
        x = self.conv_t2(x)
        x = self.bn2(x)
        outputs = self.output_conv(x)
        return outputs

class DVAE(Model):
    """The complete VAE model, with encoder and decoder."""
    def __init__(self, latent_dim, num_colors, grid_size, **kwargs):
        super(DVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_colors = num_colors
        self.grid_size = grid_size
        self.encoder = Encoder(latent_dim=latent_dim, grid_size=grid_size)
        self.decoder = Decoder(latent_dim=latent_dim, num_colors=num_colors, grid_size=grid_size)
        self.sampling = Sampling()

    def call(self, inputs):
        inputs_f32 = tf.cast(inputs, dtype=tf.float32)
        z_mean, z_log_var = self.encoder(inputs_f32)
        z = self.sampling([z_mean, z_log_var])
        reconstruction = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        self.add_loss(tf.reduce_mean(kl_loss))
        return reconstruction

class ModelTrainer:
    """Handles the compilation and training of the VAE model."""
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.checkpoint_path = os.path.join(self.config.vae_checkpoints_dir, 'dvae_checkpoint.weights.h5')

    def train(self):
        logging.info("Compiling VAE model...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(from_logits=False), run_eagerly=False)
        logging.info("Starting VAE training...")
        callbacks = [
            ModelCheckpoint(self.checkpoint_path, save_best_only=True, save_weights_only=True),
            TensorBoard(log_dir=os.path.join(self.config.project_root, 'logs')),
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        ]
        self.model.fit(
            self.dataset,
            epochs=self.config.epochs,
            callbacks=callbacks
        )
        self.model.save_weights(self.checkpoint_path)
        logging.info(f"D-VAE model weights saved to {self.checkpoint_path}")


class PlausibilityValidator:
    """Uses the trained VAE to score the plausibility of an ARC grid."""
    def __init__(self, vae_model):
        self.vae_model = vae_model
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=False)

    def get_plausibility_score(self, grid):
        """Calculates a plausibility score based on VAE's reconstruction error.
        Args:
            grid (np.array): A raw grid of shape (H, W) with integer color values (0-9).
        Returns:
            float: A score between 0 and 1, where 1 is perfectly plausible.
        """
        grid_tensor_input = tf.cast(tf.expand_dims(tf.expand_dims(grid, axis=0), axis=-1), dtype=tf.float32)
        reconstruction = self.vae_model(grid_tensor_input, training=False)
        grid_tensor_loss_target = tf.cast(tf.expand_dims(tf.expand_dims(grid, axis=0), axis=-1), dtype=tf.int32)
        reconstruction = tf.ensure_shape(reconstruction, (1, self.vae_model.grid_size, self.vae_model.grid_size, self.vae_model.num_colors))
        grid_tensor_loss_target = tf.ensure_shape(grid_tensor_loss_target, (1, self.vae_model.grid_size, self.vae_model.grid_size, 1))
        reconstruction_loss = self.loss_fn(grid_tensor_loss_target, reconstruction)
        return tf.exp(-reconstruction_loss).numpy()

class RLAgent:
    """
    A Reinforcement Learning agent using a Deep Q-Network (DQN) to generate
    plausible ARC grids based on feedback from a trained VAE.
    """
    def __init__(self, vae_model, grid_size, num_colors, learning_rate=0.001,
                 discount_factor=0.99, exploration_rate=1.0, min_exploration_rate=0.01,
                 exploration_decay_rate=0.995):
        self.vae_model = vae_model
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.action_space_size = grid_size * grid_size * num_colors  # Each action is (x, y, color)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        # Build the DQN (policy network)
        self.q_network = self._build_q_network()
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def _build_q_network(self):
        """
        Creates a neural network to approximate the Q-values.
        The state is the flattened grid, and the output is the Q-value for each possible action.
        """
        inputs = layers.Input(shape=(self.grid_size, self.grid_size, 1), dtype=tf.float32)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(self.action_space_size)(x)
        return Model(inputs=inputs, outputs=outputs)

    def choose_action(self, state_grid):
        """
        Chooses an action using an epsilon-greedy policy.

        Args:
            state_grid (np.array): The current state grid (H, W).
        
        Returns:
            tuple: The chosen action (x, y, color).
        """
        if np.random.uniform(0, 1) < self.exploration_rate:
            # Explore: choose a random action
            action_index = np.random.randint(self.action_space_size)
        else:
            # Exploit: choose the best action from the Q-network
            state_tensor = tf.cast(tf.expand_dims(tf.expand_dims(state_grid, 0), -1), dtype=tf.float32)
            q_values = self.q_network(state_tensor, training=False)
            action_index = tf.argmax(q_values[0]).numpy()
            
        return self._map_index_to_action(action_index)

    def _map_index_to_action(self, index):
        """Maps a flat action index to (x, y, color)."""
        color = index % self.num_colors
        index = index // self.num_colors
        y = index % self.grid_size
        x = index // self.grid_size
        return (x, y, color)

    def _map_action_to_index(self, x, y, color):
        """Maps an action (x, y, color) to a flat index."""
        return x * self.grid_size * self.num_colors + y * self.num_colors + color

    def train_step(self, state, action, reward, next_state, done):
        """
        Performs a single training step to update the Q-network.

        Args:
            state (np.array): The previous grid.
            action (tuple): The action taken (x, y, color).
            reward (float): The reward received.
            next_state (np.array): The new grid after the action.
            done (bool): Whether the episode has finished.
        """
        state_tensor = tf.cast(tf.expand_dims(tf.expand_dims(state, 0), -1), dtype=tf.float32)
        next_state_tensor = tf.cast(tf.expand_dims(tf.expand_dims(next_state, 0), -1), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Get the Q-values for the current state
            q_values = self.q_network(state_tensor)
            
            # Calculate the target Q-value
            if done:
                target_q_value = reward
            else:
                next_q_values = self.q_network(next_state_tensor)
                max_next_q_value = tf.reduce_max(next_q_values)
                target_q_value = reward + self.discount_factor * max_next_q_value

            # Create a mask for the taken action
            action_mask = tf.one_hot(self._map_action_to_index(*action), self.action_space_size, dtype=tf.float32)
            
            # Calculate the loss
            q_value_for_action = tf.reduce_sum(q_values * action_mask, axis=1)
            loss = tf.keras.losses.MSE(tf.expand_dims(target_q_value, 0), q_value_for_action)

        # Backpropagate and update the network
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        
    def get_vae_reward(self, grid):
        """
        Calculates a reward based on the VAE's reconstruction loss.
        Lower loss = higher reward.

        Args:
            grid (np.array): The grid to evaluate.
        
        Returns:
            float: The reward value.
        """
        # A simple reward is the negative of the reconstruction loss
        # The reconstruction loss is a measure of "un-learnability"
        plausibility_validator = self.vae_model
        loss = -plausibility_validator.get_plausibility_score(grid)
        return float(loss)

    def decay_exploration_rate(self):
        """
        Decays the exploration rate to balance exploration and exploitation.
        """
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)
        
    def generate_grid_with_rl(self, max_steps=1000):
        """
        Generates a plausible grid using the trained RL agent.
        """
        current_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for step in range(max_steps):
            action = self.choose_action(current_grid)
            x, y, color = action
            
            # Take the action
            next_grid = current_grid.copy()
            next_grid[x, y] = color
            
            # Get a reward from the VAE
            reward = self.get_vae_reward(next_grid)
            
            # The episode is done when the grid is full or we reach max steps
            done = step == max_steps - 1
            
            # Train the agent
            self.train_step(current_grid, action, reward, next_grid, done)
            
            current_grid = next_grid
            
        return current_grid

def main():
    """Orchestrates the data loading, model training, and validation demonstration."""
    logging.info("Initializing ARC AGI Project...")
    mount_google_drive()

    config = ARCConfig()
    os.makedirs(config.vae_checkpoints_dir, exist_ok=True)

    # 1. Data Preparation
    logging.info("Loading and preparing data...")
    arc_data_loader = ARCDataLoader(config)
    synthetic_data_loader = SyntheticDataLoader(config)

    arc_ds = arc_data_loader.load_dataset()
    synthetic_ds = synthetic_data_loader.load_dataset()

    # Limit the dataset size for debugging
    # arc_ds = arc_ds.take(100) # Take only the first 100 elements
    # synthetic_ds = synthetic_ds.take(100) # Take only the first 100 elements


    combined_ds = arc_ds.concatenate(synthetic_ds)

    # 2. VAE Model Training
    logging.info("Building VAE model...")
    # Pass grid_size to Encoder and Decoder
    vae = DVAE(latent_dim=config.latent_dim, num_colors=config.num_colors, grid_size=config.grid_size)

    # Explicitly build the model with a sample input shape.
    # This creates all the layers and their weights before training.
    vae.build(input_shape=(None, config.grid_size, config.grid_size, 1))

    trainer = ModelTrainer(config, vae, combined_ds)
    trainer.train()

    # 3. Plausibility Validator Demonstration
    logging.info("\nDemonstrating Plausibility Validator...")
    validator = PlausibilityValidator(vae)

    # Example 1: Plausible grid (a simple square of color 1)
    plausible_grid = np.zeros((config.grid_size, config.grid_size), dtype=np.int32)
    plausible_grid[5:10, 5:10] = 1
    plausibility_score_1 = validator.get_plausibility_score(plausible_grid)
    logging.info(f"Plausibility score for a simple square grid: {plausibility_score_1:.4f}")

    # Example 2: Implausible grid (random noise)
    implausible_grid = np.random.randint(0, config.num_colors, size=(config.grid_size, config.grid_size), dtype=np.int32)
    plausibility_score_2 = validator.get_plausibility_score(implausible_grid)
    logging.info(f"Plausibility score for a random noise grid: {plausibility_score_2:.4f}")

if __name__ == "__main__":
    main()






