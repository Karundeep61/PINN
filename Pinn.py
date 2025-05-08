import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Domain setup
x_min, x_max, y_min, y_max, n_pts = 0.0, 1.0, 0.0, 0.5, 200
xx = np.linspace(x_min, x_max, n_pts)
yy = np.linspace(y_min, y_max, n_pts)
X, Y = np.meshgrid(xx, yy)
XY = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float32)
XY_tf = tf.convert_to_tensor(XY)

# PINN model definition
def build_pinn():
    inp = layers.Input(shape=(2,))
    x = layers.Dense(64, activation='tanh')(inp)
    for _ in range(3):
        x = layers.Dense(64, activation='tanh')(x)
    out = layers.Dense(3, activation=None)(x)  # u, v, p
    return models.Model(inp, out)

model = build_pinn()

# Compute Navier-Stokes residuals using persistent tapes
def ns_residuals(xy, model):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xy)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(xy)
            uvp = model(xy)
            u = uvp[:, 0:1]
            v = uvp[:, 1:2]
            p = uvp[:, 2:3]
        # First derivatives of u, v, and p
        grads_u = tape1.gradient(u, xy)
        u_x = grads_u[:, 0:1]
        u_y = grads_u[:, 1:2]
        grads_v = tape1.gradient(v, xy)
        v_x = grads_v[:, 0:1]
        v_y = grads_v[:, 1:2]
        grads_p = tape1.gradient(p, xy)
        p_x = grads_p[:, 0:1]
        p_y = grads_p[:, 1:2]
    # Second derivatives for u and v
    u_xx = tape2.gradient(u_x, xy)[:, 0:1]
    u_yy = tape2.gradient(u_y, xy)[:, 1:2]
    v_xx = tape2.gradient(v_x, xy)[:, 0:1]
    v_yy = tape2.gradient(v_y, xy)[:, 1:2]
    # Clean up tapes
    del tape1, tape2

    # Physical parameters
    rho = 1.0
    nu = 0.01
    # Continuity equation residual
    cont = u_x + v_y
    # Momentum equation residuals
    mom_u = u * u_x + v * u_y + (1.0 / rho) * p_x - nu * (u_xx + u_yy)
    mom_v = u * v_x + v * v_y + (1.0 / rho) * p_y - nu * (v_xx + v_yy)
    return cont, mom_u, mom_v

# Training setup
def train_step(xy):
    with tf.GradientTape() as tape:
        cont_r, mu_r, mv_r = ns_residuals(xy, model)
        loss = tf.reduce_mean(cont_r**2) + tf.reduce_mean(mu_r**2) + tf.reduce_mean(mv_r**2)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Training loop
epochs = 2000
for epoch in range(epochs):
    loss_value = train_step(XY_tf)
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss_value.numpy():.6f}")

# Prediction and visualization
uvp_pred = model(XY_tf).numpy()
U = uvp_pred[:, 0].reshape(n_pts, n_pts)
V = uvp_pred[:, 1].reshape(n_pts, n_pts)

plt.figure(figsize=(6, 4))
plt.streamplot(X, Y, U, V, density=1.2)
plt.title("Predicted 2D Flow Field")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Save model
model.save("pinn_aero_ns_model.h5")
model.save("pinn_aero_ns_model.h5")
