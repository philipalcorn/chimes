import torch
import matplotlib.pyplot as plt   

# Make synthetic data from a known line
true_m, true_b = 2.5, -1

x=torch.linspace(-3,3,100) # 100 inputs
noise = 0.5*torch.randn(100) # Gaussian noise
y_true = true_m * x + true_b + noise # targets: y =~ 2.5x - 1 + noise

# Parameters we want to learn (starting at garbage values)
m = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

lr = 0.01
for epoch in range(200):
    y = m * x + b                  # foward pass over all 100 points
    loss = ((y -y_true) ** 2).mean()    # mean squared error, a scalar

    loss.backward()
    with torch.no_grad():
        m -= lr * m.grad                 # update step, not part of graph
        b -= lr * b.grad
        m.grad.zero_()                  # clear gradients before next iter
        b.grad.zero_()

    if epoch % 40 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.3f} | "
              f"m {m.item():.3f} | b {b.item():.3f}")
print(f"\nlearned: m={m.item():.3f}, b={b.item():.3f}")
print(f"true:    m={true_m}, b={true_b}")


# Plotting data and results --- at this point the "model" is trained!!!
with torch.no_grad():
    y_pred = m * x + b                      # the fitted line, untracked

plt.scatter(x, y_true, s=18, alpha=0.6, label='data')
plt.plot(x, y_pred, 'r-', lw=2,
         label=f'model: y = {m.item():.2f}x + {b.item():.2f}')
plt.xlabel('x'); plt.ylabel('y')
plt.legend()
plt.show()



# Trying the model on new data
# Test the model on new data
x_new = torch.rand((1,100)) * 50
with torch.no_grad():
    y_new = m * x_new + b

plt.scatter(x, y_true, s=18, alpha=0.6, label='training data')
plt.scatter(x_new, y_new, color='red', s=60, zorder=5, label='new predictions')
plt.xlabel('x'); plt.ylabel('y')
plt.legend()
plt.show()
