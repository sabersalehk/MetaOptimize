
from typing import Any, Callable, NamedTuple, Optional, Union
import chex
import jax
import jax.numpy as jnp
import optax


def MetaOptimize_AdamW_Lion_hg(
    meta_stepsize: float = 1e-3,
    alpha0: float = 1e-3,   # conservative initialization for Adam
    gamma = 0.9999, 
    b1: float = 0.9,        # base-adamw parameter
    b2: float = 0.999,      # base-adamw parameters
    meta_b1: float = 0.99,  # meta-lion parameters
    meta_c1: float = 0.9,   # meta-lion parameters
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[optax.Params],
                                                    Any]]] = None,
) -> optax.GradientTransformation:

    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)
        nu = jax.tree_map(jnp.zeros_like, params)
        h = jax.tree_map(jnp.zeros_like, params)
        beta = jnp.array(jnp.log(alpha0), dtype=jnp.float32)
        meta_momentum = jnp.zeros([], jnp.float32)
        return ScaleByState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, h=h, beta=beta, meta_momentum=meta_momentum)
    
    def update_fn(grads, state, params):
        eps = 1e-8
        count = state.count + 1

        # base update AdamW
        mu = _update_moment(grads, state.mu, b1, 1)
        nu = _update_moment(grads, state.nu, b2, 2)

        mu_hat = _bias_correction(mu, b1, count)
        nu_hat = _bias_correction(nu, b2, count)

        updates = normalize(mu_hat, nu_hat, eps)
        updates = add_decayed_weights(updates, params, weight_decay, weight_decay_mask)
    
        meta_grad = inner_product(state.h, grads)
        
        #meta Lion:
        meta_momentum = _update_moment_scalar(meta_grad, state.meta_momentum, meta_b1, 1)
        meta_update = jnp.sign(meta_c1*meta_momentum + (1-meta_c1)*meta_grad)
        
        beta = state.beta + meta_stepsize * meta_update
        alpha = jnp.exp(beta)
        updates = jax.tree_map(lambda u: -alpha * u, updates)  # negate because Optax expects "descent"
        
        h = jax.tree_map(lambda h_,g,v: gamma*(1.0-alpha*weight_decay) * h_ + alpha*g/(jnp.sqrt(v)+eps), state.h, grads, nu_hat)
    
        new_state = ScaleByState(count=count, mu=mu, nu=nu, h=h, beta=beta, meta_momentum=meta_momentum)
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


class ScaleByState(NamedTuple):
  count: chex.Array  
  mu: optax.Updates
  nu: optax.Updates
  h: optax.Updates
  beta: chex.Array
  meta_momentum: chex.Array


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_map(
      lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments)

def _update_moment_scalar(updates, moments, decay, order):
  return decay*moments +(1-decay)*(updates**order)


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)

def add_decayed_weights(updates, params, weight_decay, weight_decay_mask):
  # Add decayed weights
  if params is not None:
    if weight_decay_mask is None:
        new_updates = jax.tree_map(lambda g, p: g + weight_decay * p, updates, params)
    else:
        mask = weight_decay_mask(params)
        new_updates = jax.tree_map(
            lambda g, p, m: g + weight_decay * p if m else g,
            updates, params, mask
        )
  return new_updates

def normalize(numerator, denum_square, eps):
  def normalize__(m, v):
      return m / (jnp.sqrt(v) + eps)
  return jax.tree_map(normalize__, numerator, denum_square)

def inner_product(tree1, tree2):
    #return sum(jnp.vdot(x, y) for x, y in jax.tree_util.tree_leaves(jax.tree_map(lambda a, b: (a, b), tree1, tree2)))
  return sum(jnp.vdot(x[0], x[1]) for x in jax.tree_util.tree_leaves(jax.tree_map(lambda a, b: (a, b), tree1, tree2)))













##########################################
###       Training on MNIST:
##########################################


import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
from typing import NamedTuple
from tensorflow.keras.datasets import mnist


HPARAMS = {
    "meta_stepsize": 0.001,
    "alpha0": 1e-3, 
    "gamma": 0.9999, 
    "b1": 0.9,
    "b2": 0.999,
    "meta_b1": .99, # for Lion
    "meta_c1": .9, # for Lion
    "weight_decay": 0.1,
}

# --- Model ---
def mlp_fn(x):
    mlp = hk.Sequential([
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(10),
    ])
    return mlp(x)

net = hk.transform(mlp_fn)


# --- State ---
class TrainState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


# --- Loss ---
def compute_loss(params, x, y):
    logits = net.apply(params, None, x)
    labels = jax.nn.one_hot(y, 10)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(labels * log_probs, axis=-1))


# --- Accuracy ---
def compute_accuracy(params, x, y):
    logits = net.apply(params, None, x)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == y)


# --- Step ---
@jax.jit
def train_step(state: TrainState, x, y, optimizer: optax.GradientTransformation):
    grads = jax.grad(compute_loss)(state.params, x, y)
    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    return TrainState(new_params, new_opt_state)


# --- Data ---
def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.
    return (x_train, y_train), (x_test, y_test)


def get_batches(x, y, batch_size):
    indices = np.random.permutation(len(x))
    for start in range(0, len(x), batch_size):
        idx = indices[start:start + batch_size]
        yield x[idx], y[idx]


# --- Main ---
def main():
    rng = jax.random.PRNGKey(0)
    (x_train, y_train), (x_test, y_test) = load_dataset()
    x_train, y_train = jnp.array(x_train), jnp.array(y_train)
    x_test, y_test = jnp.array(x_test), jnp.array(y_test)

    # Initialize model
    sample_input = jnp.zeros((1, 784), jnp.float32)
    params = net.init(rng, sample_input)

    # Initialize optimizer
    optimizer = MetaOptimize_AdamW_Lion_hg(**HPARAMS)
    opt_state = optimizer.init(params)
    state = TrainState(params, opt_state)

    # Training
    epochs = 10
    batch_size = 128

    for epoch in range(1, epochs + 1):
        for xb, yb in get_batches(x_train, y_train, batch_size):
            xb, yb = jnp.array(xb), jnp.array(yb)
            state = train_step(state, xb, yb, optimizer)

        train_acc = compute_accuracy(state.params, x_train, y_train)
        test_acc = compute_accuracy(state.params, x_test, y_test)

        print(f"Epoch {epoch:02d} | Train Acc: {train_acc * 100:.2f}% | Test Acc: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()


