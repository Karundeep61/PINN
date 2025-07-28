import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# PINN CLASS
# =============================================================================
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
# DRIVER CODE FOR STENOSED ARTERY BLOOD FLOW
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    # Physical properties of blood (approximations)
    blood_rho = 1060.0  # kg/m^3
    blood_nu = 0.0035 / blood_rho # m^2/s (kinematic viscosity)
    
    # Geometry of the artery
    L = 0.1  # Artery length (m)
    H = 0.01 # Artery height (m)
    stenosis_L = 0.02 # Length of the stenosis section
    stenosis_H = 0.003 # Height of the stenosis bump (controls severity)
    
    # Pulsatile flow properties
    U_mean = 0.2 # Mean velocity (m/s)
    heart_rate_hz = 1.2 # Heart rate in Hz (72 bpm)

    # --- Model and Optimizer ---
    layers = [3, 50, 50, 50, 50, 3]
    pinn = PINN(layers, rho=blood_rho, nu=blood_nu).to(device)
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # --- Data Generation for Stenosed Artery ---
    # Helper functions for geometry
    def get_wall_y(x):
        # A smooth bump for the stenosis
        bump = stenosis_H * (1 + np.cos(2 * np.pi * (x - L/2) / stenosis_L)) / 2
        # Apply bump only in the stenosis region
        is_in_stenosis = (x >= (L/2 - stenosis_L/2)) & (x <= (L/2 + stenosis_L/2))
        return H/2 - is_in_stenosis * bump, -H/2 + is_in_stenosis * bump
    
    # 1. PDE Collocation points (using rejection sampling)
    num_pde_points = 20000
    t_pde = torch.rand((num_pde_points, 1))
    x_pde = torch.rand((num_pde_points, 1)) * L
    y_pde = (torch.rand((num_pde_points, 1)) - 0.5) * H
    
    y_wall_upper, y_wall_lower = get_wall_y(x_pde.numpy())
    valid_indices = (y_pde.numpy() <= y_wall_upper) & (y_pde.numpy() >= y_wall_lower)
    # Filter the tensors
    t_pde = t_pde[valid_indices]
    x_pde = x_pde[valid_indices]
    y_pde = y_pde[valid_indices]
    # Reshape back to column vectors and send to device
    t_pde, x_pde, y_pde = t_pde.unsqueeze(1).to(device), x_pde.unsqueeze(1).to(device), y_pde.unsqueeze(1).to(device)

    # 2. Initial Condition (IC) points at t=0 (fluid at rest)
    num_ic_points = 5000
    t_ic = torch.zeros((num_ic_points, 1))
    x_ic = torch.rand((num_ic_points, 1)) * L
    y_ic = (torch.rand((num_ic_points, 1)) - 0.5) * H
    y_wall_upper_ic, y_wall_lower_ic = get_wall_y(x_ic.numpy())
    valid_indices_ic = (y_ic.numpy() <= y_wall_upper_ic) & (y_ic.numpy() >= y_wall_lower_ic)
    # Filter the tensors
    t_ic = t_ic[valid_indices_ic]
    x_ic = x_ic[valid_indices_ic]
    y_ic = y_ic[valid_indices_ic]
    # Reshape back to column vectors and send to device
    t_ic, x_ic, y_ic = t_ic.unsqueeze(1).to(device), x_ic.unsqueeze(1).to(device), y_ic.unsqueeze(1).to(device)
    u_ic_target = torch.zeros_like(x_ic).to(device)
    v_ic_target = torch.zeros_like(y_ic).to(device)

    # 3. Boundary Condition (BC) points
    num_bc_points = 2000
    t_bc = torch.rand((num_bc_points, 1)).to(device)

    # Inlet (x=0): Pulsatile parabolic flow (Womersley profile approximation)
    x_inlet = torch.zeros_like(t_bc).to(device)
    y_inlet = (torch.rand((num_bc_points, 1)) - 0.5) * H
    y_inlet = y_inlet.to(device)
    # Target velocity u = U(t) * (1 - (y/(H/2))^2)
    u_inlet_pulse = U_mean * (1 + torch.sin(2 * torch.pi * heart_rate_hz * t_bc))
    u_inlet_target = u_inlet_pulse * (1 - (y_inlet / (H/2))**2)
    v_inlet_target = torch.zeros_like(u_inlet_target).to(device)
    
    # Outlet (x=L): Zero pressure
    x_outlet = torch.ones_like(t_bc) * L
    y_outlet = (torch.rand((num_bc_points, 1)) - 0.5) * H
    x_outlet, y_outlet = x_outlet.to(device), y_outlet.to(device)
    p_outlet_target = torch.zeros_like(t_bc).to(device)

    # Walls (top and bottom): No-slip condition (u=0, v=0)
    x_wall = torch.rand((num_bc_points, 1)) * L
    y_wall_upper, y_wall_lower = get_wall_y(x_wall.numpy())
    x_wall, y_wall_upper, y_wall_lower = x_wall.to(device), torch.from_numpy(y_wall_upper).float().to(device), torch.from_numpy(y_wall_lower).float().to(device)
    u_wall_target = torch.zeros_like(x_wall).to(device)
    v_wall_target = torch.zeros_like(y_wall_upper).to(device)

    # --- Training Loop ---
    epochs = 10000 # More epochs might be needed for this complex problem
    start_time = time.time()
    loss_history = []
    
    print("Starting training for stenosed artery blood flow...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. PDE Loss
        loss_pde = pinn.compute_pde_loss(t_pde, x_pde, y_pde)
        
        # 2. IC Loss
        u_ic_pred, v_ic_pred, _ = pinn.forward(t_ic, x_ic, y_ic)
        loss_ic = loss_fn(u_ic_pred, u_ic_target) + loss_fn(v_ic_pred, v_ic_target)
        
        # 3. BC Loss
        # Inlet
        u_inlet_pred, v_inlet_pred, _ = pinn.forward(t_bc, x_inlet, y_inlet)
        loss_inlet = loss_fn(u_inlet_pred, u_inlet_target) + loss_fn(v_inlet_pred, v_inlet_target)
        # Outlet
        _, _, p_outlet_pred = pinn.forward(t_bc, x_outlet, y_outlet)
        loss_outlet = loss_fn(p_outlet_pred, p_outlet_target)
        # Walls
        u_wall_upper_pred, v_wall_upper_pred, _ = pinn.forward(t_bc, x_wall, y_wall_upper)
        u_wall_lower_pred, v_wall_lower_pred, _ = pinn.forward(t_bc, x_wall, y_wall_lower)
        loss_wall = (loss_fn(u_wall_upper_pred, u_wall_target) + loss_fn(v_wall_upper_pred, v_wall_target) +
                     loss_fn(u_wall_lower_pred, u_wall_target) + loss_fn(v_wall_lower_pred, v_wall_target))

        loss_bc = loss_inlet + loss_outlet + loss_wall
        
        # Total Loss (with weights to prioritize boundary conditions)
        total_loss = loss_pde + loss_ic + 10 * loss_bc
        
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())
        
        if (epoch + 1) % 500 == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.6f}, Time: {elapsed_time:.2f}s")
    
    print("Training finished.")

    # --- Plotting Loss Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title('Training Loss Curve (Blood Flow)')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.grid(True)
    plt.show()

    # --- Visualization ---
    print("Generating visualization...")
    pinn.eval()
    
    # Visualize at peak systole (when inlet velocity is highest)
    t_vis_val = 0.25 / heart_rate_hz 
    t_vis = torch.ones(100 * 100, 1) * t_vis_val
    
    x_vis = torch.linspace(0, L, 100)
    y_vis = torch.linspace(-H/2, H/2, 100)
    x_grid, y_grid = torch.meshgrid(x_vis, y_vis, indexing='xy')
    
    t_flat = t_vis.to(device)
    x_flat = x_grid.flatten().unsqueeze(1).to(device)
    y_flat = y_grid.flatten().unsqueeze(1).to(device)

    with torch.no_grad():
        u_pred, v_pred, p_pred = pinn.forward(t_flat, x_flat, y_flat)

    # Mask out points outside the artery for plotting
    y_wall_upper_vis, y_wall_lower_vis = get_wall_y(x_flat.cpu().numpy())
    invalid_indices_vis = (y_flat.cpu().numpy() > y_wall_upper_vis) | (y_flat.cpu().numpy() < y_wall_lower_vis)
    u_pred[invalid_indices_vis] = torch.nan
    v_pred[invalid_indices_vis] = torch.nan

    u_pred_grid = u_pred.cpu().reshape(100, 100)
    v_pred_grid = v_pred.cpu().reshape(100, 100)
    
    plt.figure(figsize=(15, 4))
    speed = torch.sqrt(u_pred_grid**2 + v_pred_grid**2)
    plt.streamplot(x_grid.numpy(), y_grid.numpy(), u_pred_grid.numpy(), v_pred_grid.numpy(), color=speed.numpy(), cmap='viridis', density=2.0)
    plt.colorbar(label='Velocity Magnitude (m/s)')
    
    # Plot artery walls
    x_wall_plot = np.linspace(0, L, 200)
    y_wall_upper_plot, y_wall_lower_plot = get_wall_y(x_wall_plot)
    plt.plot(x_wall_plot, y_wall_upper_plot, 'k-', linewidth=2)
    plt.plot(x_wall_plot, y_wall_lower_plot, 'k-', linewidth=2)
    plt.fill_between(x_wall_plot, y_wall_upper_plot, y_wall_lower_plot, color='k', alpha=0.1)

    plt.title(f'Blood Flow Velocity in Stenosed Artery at t={t_vis_val:.2f}s ❤️')
    plt.xlabel('Artery Length (m)')
    plt.ylabel('Artery Height (m)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()