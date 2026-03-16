class RLAgent:
    """
    A Reinforcement Learning agent using a Deep Q-Network (DQN) to generate
    plausible ARC grids based on feedback from a trained VAE.
    """
    def __init__(self, vae_model, plausibility_validator, grid_size, num_colors, learning_rate=0.001,
                 discount_factor=0.99, exploration_rate=1.0, min_exploration_rate=0.01,
                 exploration_decay_rate=0.995):
        self.vae_model = vae_model
        self.plausibility_validator = plausibility_validator
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.action_space_size = grid_size * grid_size * num_colors
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_network = self._build_q_network()
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def _build_q_network(self):
        inputs = layers.Input(shape=(self.grid_size, self.grid_size, 1), dtype=tf.float32)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(self.action_space_size)(x)
        return Model(inputs=inputs, outputs=outputs)

    def choose_action(self, state_grid):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action_index = np.random.randint(self.action_space_size)
        else:
            state_tensor = tf.cast(tf.expand_dims(tf.expand_dims(state_grid, 0), -1), dtype=tf.float32)
            q_values = self.q_network(state_tensor, training=False)
            action_index = tf.argmax(q_values[0]).numpy()
        return action_index

    def _map_index_to_action(self, index):
        """Maps a linear action index to (x, y, color)."""
        color = index % self.num_colors
        index //= self.num_colors
        y = index % self.grid_size
        x = index // self.grid_size
        return (x, y, color)

    def _map_action_to_index(self, x, y, color):
        """Maps (x, y, color) to a linear action index."""
        return x * self.grid_size * self.num_colors + y * self.num_colors + color

    @tf.function
    def train_step(self, state, action_index, reward, next_state, done):
        state_tensor = tf.cast(tf.expand_dims(tf.expand_dims(state, 0), -1), dtype=tf.float32)
        next_state_tensor = tf.cast(tf.expand_dims(tf.expand_dims(next_state, 0), -1), dtype=tf.float32)

        with tf.GradientTape() as tape:
            q_values = self.q_network(state_tensor)

            if done:
                target_q_value = reward
            else:
                next_q_values = self.q_network(next_state_tensor)
                max_next_q_value = tf.reduce_max(next_q_values)
                target_q_value = reward + self.discount_factor * max_next_q_value

            # --- FIX 2 (cont.): No longer need to map action to index here ---
            action_mask = tf.one_hot(action_index, self.action_space_size, dtype=tf.float32)
            q_value_for_action = tf.reduce_sum(q_values * action_mask, axis=1)
            loss = tf.keras.losses.MSE(tf.expand_dims(target_q_value, 0), q_value_for_action)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def get_vae_reward(self, grid):
        plausibility_score = self.plausibility_validator.get_plausibility_score(grid)
        return float(plausibility_score)

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)

    def generate_grid_with_rl(self, max_steps=1000):
        logging.info("RL Agent: Starting grid generation...")
        current_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        current_grid[0,0] = 1

        for step in range(max_steps):
            action_index = self.choose_action(current_grid)
            x, y, color = self._map_index_to_action(action_index)

            next_grid = current_grid.copy()
            next_grid[x, y] = color

            reward = self.get_vae_reward(next_grid)
            done = step == max_steps - 1

            self.train_step(current_grid, action_index, reward, next_grid, done)

            current_grid = next_grid
            self.decay_exploration_rate()

            if step % 100 == 0:
                logging.info(f"RL Agent: Step {step}/{max_steps}, Reward: {reward:.4f}, Exploration: {self.exploration_rate:.4f}")

        logging.info("RL Agent: Grid generation complete.")
        return current_grid
