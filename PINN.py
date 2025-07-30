import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np # NEW: Import numpy

# --- (The PINN class definition is identical) ---
class PINN(nn.Module):
    def __init__(self, layers, rho=1.0, nu=0.01):
        super().__init__()
        self.rho = rho
        self.nu = nu
        self.net = self.build_net(layers)

    def build_net(self, layers):
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        return nn.Sequential(*modules)

    def forward(self, t, x, y):
        inputs = torch.cat([t, x, y], dim=1)
        output = self.net(inputs)
        u = output[:, 0:1]
        v = output[:, 1:2]
        p = output[:, 2:3]
        return u, v, p

    def compute_pde_loss(self, t, x, y):
        t.requires_grad_(True)
        x.requires_grad_(True)
        y.requires_grad_(True)
        u, v, p = self.forward(t, x, y)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        continuity_eq = u_x + v_y
        x_momentum_eq = u_t + u * u_x + v * u_y + (1/self.rho) * p_x - self.nu * (u_xx + u_yy)
        y_momentum_eq = v_t + u * v_x + v * v_y + (1/self.rho) * p_y - self.nu * (v_xx + v_yy)
        loss_fn = nn.MSELoss()
        pde_loss = loss_fn(continuity_eq, torch.zeros_like(continuity_eq)) + \
                   loss_fn(x_momentum_eq, torch.zeros_like(x_momentum_eq)) + \
                   loss_fn(y_momentum_eq, torch.zeros_like(y_momentum_eq))
        return pde_loss


# =============================================================================
# DRIVER CODE
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    layers = [3, 40, 40, 40, 40, 3]
    pinn = PINN(layers, rho=1.0, nu=0.01).to(device)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # --- (Data Generation is identical) ---
    num_pde_points = 10000
    t_pde = torch.rand((num_pde_points, 1)).to(device)
    x_pde = torch.rand((num_pde_points, 1)).to(device)
    y_pde = torch.rand((num_pde_points, 1)).to(device)
    num_ic_points = 2500
    t_ic = torch.zeros((num_ic_points, 1)).to(device)
    x_ic = torch.rand((num_ic_points, 1)).to(device)
    y_ic = torch.rand((num_ic_points, 1)).to(device)
    u_ic_target = torch.zeros_like(x_ic).to(device)
    v_ic_target = torch.zeros_like(y_ic).to(device)
    num_bc_points = 1000
    t_bc = torch.rand((num_bc_points, 1)).to(device)
    x_top = torch.rand((num_bc_points, 1)).to(device)
    y_top = torch.ones_like(x_top).to(device)
    u_top_target = torch.ones_like(x_top).to(device)
    v_top_target = torch.zeros_like(x_top).to(device)
    x_bottom = torch.rand((num_bc_points, 1)).to(device)
    y_bottom = torch.zeros_like(x_bottom).to(device)
    y_left = torch.rand((num_bc_points, 1)).to(device)
    x_left = torch.zeros_like(y_left).to(device)
    y_right = torch.rand((num_bc_points, 1)).to(device)
    x_right = torch.ones_like(y_right).to(device)
    u_stationary_target = torch.zeros((num_bc_points, 1)).to(device)
    v_stationary_target = torch.zeros((num_bc_points, 1)).to(device)

    # --- Training Loop ---
    epochs = 5000
    start_time = time.time()
    loss_history = []
    
    print("Starting GPU-accelerated training for Lid-Driven Cavity...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_pde = pinn.compute_pde_loss(t_pde, x_pde, y_pde)
        u_ic_pred, v_ic_pred, _ = pinn.forward(t_ic, x_ic, y_ic)
        loss_ic = loss_fn(u_ic_pred, u_ic_target) + loss_fn(v_ic_pred, v_ic_target)
        u_top_pred, v_top_pred, _ = pinn.forward(t_bc, x_top, y_top)
        loss_top = loss_fn(u_top_pred, u_top_target) + loss_fn(v_top_pred, v_top_target)
        u_bottom_pred, v_bottom_pred, _ = pinn.forward(t_bc, x_bottom, y_bottom)
        loss_bottom = loss_fn(u_bottom_pred, u_stationary_target) + loss_fn(v_bottom_pred, v_stationary_target)
        u_left_pred, v_left_pred, _ = pinn.forward(t_bc, x_left, y_left)
        loss_left = loss_fn(u_left_pred, u_stationary_target) + loss_fn(v_left_pred, v_stationary_target)
        u_right_pred, v_right_pred, _ = pinn.forward(t_bc, x_right, y_right)
        loss_right = loss_fn(u_right_pred, u_stationary_target) + loss_fn(v_right_pred, v_stationary_target)
        loss_bc = loss_top + loss_bottom + loss_left + loss_right
        total_loss = loss_pde + loss_ic + loss_bc
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())
        if (epoch + 1) % 250 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss.item():.6f}, Time: {elapsed_time:.2f}s")
    
    print("Training finished.")

    # --- Plotting Loss Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.grid(True)
    plt.show()

    # --- Visualization ---
    print("Generating visualization...")
    # Set model to evaluation mode
    pinn.eval()
    
    # Create a grid of points to plot the solution
    n_points = 100
    t_vis = torch.ones(n_points * n_points, 1) # Visualize at t=1.0
    x_vis = torch.linspace(0, 1, n_points)
    y_vis = torch.linspace(0, 1, n_points)
    x_grid, y_grid = torch.meshgrid(x_vis, y_vis, indexing='xy')
    
    # Flatten grid and move to device
    t_flat = t_vis.to(device)
    x_flat = x_grid.flatten().unsqueeze(1).to(device)
    y_flat = y_grid.flatten().unsqueeze(1).to(device)

    # Get predictions
    with torch.no_grad(): # Disable gradient calculation for inference
        u_pred, v_pred, p_pred = pinn.forward(t_flat, x_flat, y_flat)

    # Reshape predictions to match grid and move to CPU for plotting
    u_pred_grid = u_pred.cpu().reshape(n_points, n_points)
    v_pred_grid = v_pred.cpu().reshape(n_points, n_points)
    p_pred_grid = p_pred.cpu().reshape(n_points, n_points)
    
    # Plotting the velocity field using streamplot
    plt.figure(figsize=(8, 8))
    speed = torch.sqrt(u_pred_grid**2 + v_pred_grid**2)
    plt.streamplot(x_grid.numpy(), y_grid.numpy(), u_pred_grid.numpy(), v_pred_grid.numpy(), color=speed.numpy(), cmap='viridis', density=1.5)
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Velocity Field of Lid-Driven Cavity 🌪️')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()