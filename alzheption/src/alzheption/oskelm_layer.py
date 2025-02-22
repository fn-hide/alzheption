import tensorflow as tf
from alzheption.lib.TfELM.Layers.KELMLayer import KELMLayer
from alzheption.lib.TfELM.Resources.kernel_distances import calculate_pairwise_distances, calculate_pairwise_distances_vector



class OSKELMLayer(KELMLayer):
    def __init__(self, kernel, activation):
        super().__init__(self, kernel, activation)
    
    def fit(self, x, y, batch_size=100):
        """
        Fit the layer to the input-output pairs using batches for kernel computation.

        Args:
        -----------
            x (tensor): Input data.
            y (tensor): Target values.
            batch_size (int): Size of each batch for kernel computation.
        """
        x = tf.cast(x, dtype=tf.float32)
        self.input = x
        n_samples = int(self.random_pct * x.shape[0])

        if self.nystrom_approximation:
            if self.landmark_selection_method not in ["stratified", "information_gain_based"]:
                L = eval(f"{self.landmark_selection_method}_sampling(x, n_samples)")
            else:
                y_new = tf.argmax(y, axis=1)
                y_new = tf.cast(y_new, dtype=tf.int32)
                L = eval(f"{self.landmark_selection_method}_sampling(x, y_new, n_samples)")
            
            # Compute C in batches
            C_batches = []
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i+batch_size]
                C_batch = calculate_pairwise_distances_vector(x_batch, L, self.kernel.ev)
                C_batches.append(C_batch)
            C = tf.concat(C_batches, axis=0)

            W = calculate_pairwise_distances(L, self.kernel.ev)
            diagonal = tf.linalg.diag_part(W)
            diagonal_with_small_value = diagonal + 0.00001
            W = tf.linalg.set_diag(W, diagonal_with_small_value)
            K = tf.matmul(tf.matmul(C, tf.linalg.inv(W)), C, transpose_b=True)
        else:
            # Compute K in batches
            K_batches = []
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i+batch_size]
                K_batch = calculate_pairwise_distances(x_batch, self.kernel.ev)
                K_batches.append(K_batch)
            K = tf.concat(K_batches, axis=0)

        diagonal = tf.linalg.diag_part(K)
        diagonal_with_small_value = diagonal + 0.1
        K = tf.linalg.set_diag(K, diagonal_with_small_value)
        self.K = tf.linalg.inv(K)
        self.beta = tf.matmul(self.K, y)
    