# ------------------------------------------------------
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import jax, jax.numpy as jnp
from flax import nnx
import optax
from functools import partial

digits = load_digits()

splits = train_test_split(digits.images / 16 > 0.5, random_state=0)

images_train, images_test = map(partial(jnp.asarray, dtype=jnp.int8), splits)

print(f"{images_train.shape=}")
print(f"{images_test.shape=}")

# ------------------------------------------------------


class Encoder(nnx.Module):
    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        self.linear = nnx.Linear(input_size, intermediate_size, rngs=rngs)
        self.linear_mean = nnx.Linear(intermediate_size, output_size, rngs=rngs)
        self.linear_std = nnx.Linear(intermediate_size, output_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        x = self.linear(x)
        x = jax.nn.relu(x)

        mean = self.linear_mean(x)
        std = jnp.exp(self.linear_std(x))

        key = self.rngs.noise()
        z = mean + std * jax.random.normal(key, mean.shape)
        return z, mean, std


class Decoder(nnx.Module):
    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear1 = nnx.Linear(input_size, intermediate_size, rngs=rngs)
        self.linear2 = nnx.Linear(intermediate_size, output_size, rngs=rngs)

    def __call__(self, z: jax.Array) -> jax.Array:
        z = self.linear1(z)
        z = jax.nn.relu(z)
        logits = self.linear2(z)
        return logits


class VAE(nnx.Module):
    def __init__(
        self,
        image_shape: tuple[int, int],
        hidden_size: int,
        latent_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.image_shape = image_shape
        self.latent_size = latent_size
        input_size = image_shape[0] * image_shape[1]
        self.encoder = Encoder(input_size, hidden_size, latent_size, rngs=rngs)
        self.decoder = Decoder(latent_size, hidden_size, input_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        x = jax.vmap(jax.numpy.ravel)(x)  # flatten
        z, mean, std = self.encoder(x)
        logits = self.decoder(z)
        logits = jnp.reshape(logits, (-1, *self.image_shape))
        return logits, mean, std


def vae_loss(model: VAE, x: jax.Array):
    logits, mean, std = model(x)

    kl_loss = jnp.mean(
        0.5 * jnp.mean(-jnp.log(std**2) - 1.0 + std**2 + mean**2, axis=-1)
    )
    reconstruction_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, x))
    return reconstruction_loss + 0.1 * kl_loss


# ------------------------------------------------------

model = VAE(
    image_shape=(8, 8),
    hidden_size=32,
    latent_size=8,
    rngs=nnx.Rngs(0, noise=1),
)

optimizer = nnx.Optimizer(model, optax.adam(1e-3))


@nnx.jit
def train_step(model: VAE, optimizer: nnx.Optimizer, x: jax.Array):
    loss, grads = nnx.value_and_grad(vae_loss)(model, x)
    optimizer.update(grads)
    return loss


# with jax.debug_nans(True):
for epoch in range(2001):
    loss = train_step(model, optimizer, images_train)
    if epoch % 500 == 0:
        print(f"Epoch {epoch} loss: {loss}")


# --------------------------------------------
logits, _, _ = model(images_test)
images_pred = jax.nn.sigmoid(logits)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(
    2,
    10,
    figsize=(6, 1.5),
    subplot_kw={"xticks": [], "yticks": []},
    gridspec_kw=dict(hspace=0.1, wspace=0.1),
)
for i in range(10):
    ax[0, i].imshow(images_test[i], cmap="binary", interpolation="gaussian")
    ax[1, i].imshow(images_pred[i], cmap="binary", interpolation="gaussian")

plt.savefig("plots/vae_Test.png")

# --------------------------------------------
import numpy as np

# generate new images by sampling the latent space
z = np.random.normal(scale=1.5, size=(36, model.latent_size))
logits = model.decoder(z).reshape(-1, 8, 8)
images_gen = nnx.sigmoid(logits)

fig, ax = plt.subplots(
    6,
    6,
    figsize=(4, 4),
    subplot_kw={"xticks": [], "yticks": []},
    gridspec_kw=dict(hspace=0.1, wspace=0.1),
)
for i in range(36):
    ax.flat[i].imshow(images_gen[i], cmap="binary", interpolation="gaussian")

plt.savefig("plots/vae_Generate.png")

# --------------------------------------------
z, _, _ = model.encoder(images_train.reshape(-1, 64))
zrange = jnp.linspace(z[2], z[9], 10)

logits = model.decoder(zrange).reshape(-1, 8, 8)
images_gen = nnx.sigmoid(logits)

fig, ax = plt.subplots(
    1,
    10,
    figsize=(8, 1),
    subplot_kw={"xticks": [], "yticks": []},
    gridspec_kw=dict(hspace=0.1, wspace=0.1),
)
for i in range(10):
    ax.flat[i].imshow(images_gen[i], cmap="binary", interpolation="gaussian")

plt.savefig("plots/vae_Interpolate.png")
