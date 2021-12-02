import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp
from tqdm import tqdm

from donk.policy.lin_gaussian import LinearGaussianPolicy
from donk.policy.policy import Policy


class Neural_Network_Policy(Policy):
    """Neural network state-action mapping policy."""

    def __init__(self, model, normalize_states=True):
        """Initialize this policy.

        Args:
            model: A `tf.keras.Model`
            normalize_states: Adds a `Normalization` layer directly after the input of the model.

        """
        # Extract dimensions
        self.dX = model.input.shape[-1]
        self.dU = model.output.shape[-1]

        # Add state normalization to model
        if normalize_states:
            self.state_normalization = layers.Normalization(name="normalized_state")

            state = tf.keras.Input((self.dX, ), name="state")
            normalized_state = self.state_normalization(state)
            action = model(normalized_state)

            self.model = tf.keras.Model(inputs=state, outputs=action)
        else:
            self.state_normalization = None
            self.model = model

        # Add metrics to model
        self.model.metric_loss_kl = tf.keras.metrics.Mean("train/loss_kl")
        self.model.metric_loss_reg = tf.keras.metrics.Mean("train/loss_reg")
        self.model.metric_loss = tf.keras.metrics.Mean("train/loss")
        self.model.metric_loss_val = tf.keras.metrics.Mean("val/loss")

    def update(self, X_train, U_train, prc_train, epochs: int, batch_size: int, X_val=None, U_val=None, prc_val=None, silent: bool = False):
        """Train the model on new data.

        May supply validation data.

        Args:
            X_train: (N_train, dX) Training states
            U_train: (N_train, dU) Training target actions
            prc_train: (N_train, dU, dU) Training action precisions
            epochs: Number of epochs
            batch_size: Batch size for training
            X_val: (N_val, dX) Validation states (optional)
            U_val: (N_val, dU) Validation target actions (optional)
            prc_val: (N_val, dU, dU) Validation action precisions (optional)
            silent: Whether to disable progress bar
        """
        N_train, _ = X_train.shape

        # Normalize states
        if self.state_normalization is not None:
            self.state_normalization.adapt(X_train)

        # Normalize precision
        prc_scale = np.einsum("ijj->", prc_train) / N_train / self.dU
        prc_train = prc_train / prc_scale
        if prc_val is not None:
            prc_val = prc_val / prc_scale

        # Build dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                X_train.astype(np.float32, copy=False),
                U_train.astype(np.float32, copy=False),
                prc_train.astype(np.float32, copy=False),
            )
        ).shuffle(N_train).batch(batch_size)
        if X_val is not None:
            dataset_test = tf.data.Dataset.from_tensor_slices(
                (
                    X_val.astype(np.float32, copy=False),
                    U_val.astype(np.float32, copy=False),
                    prc_val.astype(np.float32, copy=False),
                )
            ).batch(256)

        @tf.function
        def train_step(model, opt, state, action, precision):
            """Perform one train step."""
            with tf.GradientTape() as tape:
                action_pred = model(state, training=True)

                # KL divergence loss
                #  loss_kl = 1/2 delta_action^T * prc * delta_action
                delta_action = action - action_pred
                loss_kl = tf.reduce_mean(tf.einsum("in,inm,im->i", delta_action, precision, delta_action)) / 2

                # Regularization loss
                loss_reg = tf.reduce_sum(model.losses)

                # Total loss
                loss = loss_kl + loss_reg

            opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))

            # Update metrics
            self.model.metric_loss_kl(loss_kl)
            self.model.metric_loss_reg(loss_reg)
            self.model.metric_loss(loss)

        @tf.function
        def validation_step(model, state, action, precision):
            """Perform one validation step."""
            action_pred = model(state, training=False)

            delta_action = action - action_pred
            loss = tf.reduce_mean(tf.einsum("in,inm,im->i", delta_action, precision, delta_action)) / 2
            model.metric_loss_val(loss)

        # Reset optimizer
        opt = tf.keras.optimizers.Adam()

        with tqdm(range(epochs), disable=silent) as pbar:
            for epoch in pbar:
                self.model.reset_metrics()

                # Train batches
                for state, action, precision in dataset:
                    train_step(self.model, opt, state, action, precision)

                # Validation batches
                if X_val is not None:
                    for state, action, precision in dataset_test:
                        validation_step(self.model, state, action, precision)

                # Log metrics
                for metric in self.model.metrics:
                    tf.summary.scalar(metric.name, metric.result(), step=epoch)

                # Update progress bar
                pbar.set_description(
                    f"Train loss: {self.model.metric_loss.result():.6f}" +
                    (f" Val loss: {self.model.metric_loss_val.result():.6f}" if X_val is not None else "")
                )

    def act(self, x, t: int = None, noise=None):
        """Decide an action for the given state(s).

        Args:
            x: (..., dX) State(s)
            t: Timestep, ignored

        Returns:
            u: (..., dU) Selected action(s)
        """
        if noise is not None:
            raise NotImplementedError(f"Noise is not supported by {type(self)}")

        u = self.model(x.reshape(-1, self.dX).astype(np.float32, copy=False), training=False).numpy()
        return u.reshape(x.shape[:-1] + (self.dU, ))

    def linearize(self, X, regularization: float = 1e-6):
        """Compute linearization of this policy.

        Args:
            X: (N, T, dX) States

        Returns:
            pol_lin: LinearGaussianPolicy
        """
        N, T, _ = X.shape
        dX, dU = self.dX, self.dU

        x = tf.convert_to_tensor(X.reshape(N * T, dX), dtype=tf.float32)  # Flatten states
        with tf.GradientTape() as tape:
            tape.watch(x)
            U = tf.reshape(self.model(x, training=False), (N, T, dU))
            U_mean = tf.reduce_mean(U, axis=0)
            U_sum = tf.reduce_sum(U_mean, axis=0)  # Sum over timesteps as they are independend, to reduce dimension of du_dx

        # Compute and reshape jacobian
        du_dx = tape.jacobian(U_sum, x)
        du_dx = tf.reshape(du_dx, (dU, N, T, dX))
        du_dx = tf.reduce_mean(du_dx, axis=1)
        du_dx = tf.transpose(du_dx, (1, 0, 2))
        # du_dx.shape = (T, dU, dX)

        # Consutruct policy object
        K = du_dx.numpy()
        k = U_mean.numpy() - np.einsum("tux,tx->tu", K, np.mean(X, axis=0))
        pol_covar = tfp.stats.covariance(U).numpy() + np.eye(self.dU) * regularization  # Use sample covariance

        pol_lin = LinearGaussianPolicy(K, k, pol_covar)
        return pol_lin

    def load_weights(self, data_file):
        """Load the network weighs from a file.

        Args:
            data_file: File to load weights from
        """
        self.model.load_weights(data_file)

    def save_weights(self, data_file):
        """Save the network weighs in a file.

        Args:
            data_file: File to store weights to
        """
        self.model.save_weights(data_file)
