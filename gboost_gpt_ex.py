import matplotlib.pyplot as plt

# Steps for gradient boosting analogy
steps = [
    "1️⃣ First Guess: Give everyone 60 marks (big errors)",
    "2️⃣ First Fix: Add small rule (+15 if all questions attempted)",
    "3️⃣ Second Fix: Add another small rule (-5 if many skipped)",
    "4️⃣ Third Fix: Another rule to fix remaining mistakes",
    "✅ Final Prediction: Very close to actual scores"
]

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_facecolor("#f9f9f9")
fig.patch.set_facecolor("#f9f9f9")

# Plot steps
y_positions = list(range(len(steps)))[::-1]
ax.scatter([1]*len(steps), y_positions, s=500, color="#4a90e2", zorder=3)

# Add text for each step
for i, (step, y) in enumerate(zip(steps, y_positions)):
    ax.text(1.05, y, step, va="center", fontsize=11, color="#333", fontweight="bold")

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)

# Add title
ax.set_title("Gradient Boosting – Teacher Correcting Mistakes Analogy", fontsize=14, fontweight="bold", color="#2c3e50", pad=20)

plt.show()
