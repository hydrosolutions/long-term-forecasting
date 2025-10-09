import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import os

save_dir = "../monthly_forecasting_results/figures/methods"
os.makedirs(save_dir, exist_ok=True)
output = os.path.join(save_dir, "gradient_boosting.gif")

# Sample data - polynomial pattern y = 0.15*x^2 + 0.5*x + 2
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = 0.5 * X + 1 + np.random.normal(0, 0.3, len(X)) + 3 * np.sin(X * np.pi / 4)

# Reshape for sklearn
X_train = X.reshape(-1, 1)

# Build actual gradient boosted trees step by step
learning_rate = 0.7
initial_pred = np.mean(y)

# Store predictions and tree info at each step
preds = [np.full_like(y, initial_pred, dtype=float)]
trees = [None]

# Manually build gradient boosting (stage by stage)
current_pred = np.full_like(y, initial_pred, dtype=float)

for i in range(4):  # 4 boosting iterations
    # Calculate residuals
    residuals = y - current_pred

    # Fit a shallow tree to residuals
    tree = DecisionTreeRegressor(max_depth=2, random_state=42 + i)
    tree.fit(X_train, residuals)

    # Get tree prediction
    tree_pred = tree.predict(X_train)

    # Update current prediction
    current_pred = current_pred + learning_rate * tree_pred
    preds.append(current_pred.copy())

    # Extract tree structure for visualization (depth-2 tree)
    tree_obj = tree.tree_

    # Build tree structure recursively
    def extract_tree_info(node_id=0):
        if tree_obj.children_left[node_id] == -1:  # Leaf node
            return {"type": "leaf", "value": tree_obj.value[node_id][0][0]}
        else:
            return {
                "type": "split",
                "split": f"X ≤ {tree_obj.threshold[node_id]:.1f}",
                "left": extract_tree_info(tree_obj.children_left[node_id]),
                "right": extract_tree_info(tree_obj.children_right[node_id]),
            }

    trees.append(extract_tree_info())


def draw_tree(ax, tree, step):
    """Draw decision tree structure with compact, non-overlapping layout for depth-2 trees."""
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    if step == 0:
        # Initial prediction - centered and clear
        ax.text(
            5,
            5.5,
            f"{initial_pred:.2f}",
            fontsize=44,
            ha="center",
            va="center",
            color="#2563eb",
            weight="bold",
        )
        ax.text(
            5,
            3.5,
            "Initial Prediction",
            fontsize=13,
            ha="center",
            va="top",
            color="#475569",
            weight="bold",
        )
        ax.text(
            5,
            2.8,
            "(mean of y)",
            fontsize=11,
            ha="center",
            va="top",
            color="#64748b",
            style="italic",
        )
    else:
        # Draw depth-2 tree
        # Root node (level 0)
        root = mpatches.FancyBboxPatch(
            (3.5, 8.2),
            3.0,
            0.8,
            boxstyle="round,pad=0.06",
            edgecolor="#2563eb",
            facecolor="#dbeafe",
            linewidth=2.5,
        )
        ax.add_patch(root)
        ax.text(
            5, 8.6, tree["split"], ha="center", va="center", fontsize=10, weight="bold"
        )

        # Level 1 nodes
        left_node = tree["left"]
        right_node = tree["right"]

        # Left level-1 (position 1.8)
        if left_node["type"] == "leaf":
            box = mpatches.FancyBboxPatch(
                (1.0, 5.8),
                1.6,
                0.7,
                boxstyle="round,pad=0.05",
                edgecolor="#16a34a",
                facecolor="#dcfce7",
                linewidth=2,
            )
            ax.add_patch(box)
            ax.text(
                1.8,
                6.15,
                f"{left_node['value']:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                color="#16a34a",
            )
        else:
            box = mpatches.FancyBboxPatch(
                (0.8, 5.8),
                2.0,
                0.7,
                boxstyle="round,pad=0.05",
                edgecolor="#2563eb",
                facecolor="#dbeafe",
                linewidth=2,
            )
            ax.add_patch(box)
            ax.text(
                1.8,
                6.15,
                left_node["split"],
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
            )

        # Right level-1 (position 8.2)
        if right_node["type"] == "leaf":
            box = mpatches.FancyBboxPatch(
                (7.4, 5.8),
                1.6,
                0.7,
                boxstyle="round,pad=0.05",
                edgecolor="#dc2626",
                facecolor="#fee2e2",
                linewidth=2,
            )
            ax.add_patch(box)
            ax.text(
                8.2,
                6.15,
                f"{right_node['value']:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                color="#dc2626",
            )
        else:
            box = mpatches.FancyBboxPatch(
                (7.0, 5.8),
                2.0,
                0.7,
                boxstyle="round,pad=0.05",
                edgecolor="#2563eb",
                facecolor="#dbeafe",
                linewidth=2,
            )
            ax.add_patch(box)
            ax.text(
                8.2,
                6.15,
                right_node["split"],
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
            )

        # Branches from root
        ax.plot([4.3, 1.8], [8.2, 6.5], "k-", linewidth=2, color="#64748b", alpha=0.7)
        ax.plot([5.7, 8.2], [8.2, 6.5], "k-", linewidth=2, color="#64748b", alpha=0.7)

        # Level 2 leaf nodes
        if left_node["type"] == "split":
            # Left-left leaf
            box = mpatches.FancyBboxPatch(
                (0.2, 3.6),
                1.3,
                0.6,
                boxstyle="round,pad=0.04",
                edgecolor="#16a34a",
                facecolor="#dcfce7",
                linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(
                0.85,
                3.9,
                f"{left_node['left']['value']:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="#16a34a",
            )
            # Left-right leaf
            box = mpatches.FancyBboxPatch(
                (2.1, 3.6),
                1.3,
                0.6,
                boxstyle="round,pad=0.04",
                edgecolor="#16a34a",
                facecolor="#dcfce7",
                linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(
                2.75,
                3.9,
                f"{left_node['right']['value']:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="#16a34a",
            )
            # Branches
            ax.plot(
                [1.3, 0.85], [5.8, 4.2], "k-", linewidth=1.5, color="#64748b", alpha=0.6
            )
            ax.plot(
                [2.3, 2.75], [5.8, 4.2], "k-", linewidth=1.5, color="#64748b", alpha=0.6
            )

        if right_node["type"] == "split":
            # Right-left leaf
            box = mpatches.FancyBboxPatch(
                (6.5, 3.6),
                1.3,
                0.6,
                boxstyle="round,pad=0.04",
                edgecolor="#dc2626",
                facecolor="#fee2e2",
                linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(
                7.15,
                3.9,
                f"{right_node['left']['value']:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="#dc2626",
            )
            # Right-right leaf
            box = mpatches.FancyBboxPatch(
                (8.4, 3.6),
                1.3,
                0.6,
                boxstyle="round,pad=0.04",
                edgecolor="#dc2626",
                facecolor="#fee2e2",
                linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(
                9.05,
                3.9,
                f"{right_node['right']['value']:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="#dc2626",
            )
            # Branches
            ax.plot(
                [7.7, 7.15], [5.8, 4.2], "k-", linewidth=1.5, color="#64748b", alpha=0.6
            )
            ax.plot(
                [8.7, 9.05], [5.8, 4.2], "k-", linewidth=1.5, color="#64748b", alpha=0.6
            )

        # Tree label at bottom
        ax.text(
            5,
            2.5,
            f"Tree {step} (depth=2)",
            ha="center",
            fontsize=11,
            color="#1e293b",
            weight="bold",
        )
        ax.text(
            5,
            1.9,
            "Predicts residuals × α",
            ha="center",
            fontsize=9,
            color="#64748b",
            style="italic",
        )


def animate(frame):
    """Animate the gradient boosting process step by step."""
    step = frame

    # Clear all subplots
    for ax in [ax1, ax2]:
        ax.clear()

    # Left plot - Predictions and residuals
    ax1.scatter(
        X,
        y,
        s=100,
        c="#3b82f6",
        edgecolors="#1e40af",
        linewidth=2.5,
        zorder=3,
        label="Actual Data",
        alpha=0.9,
    )

    # Set limits with padding
    x_min, x_max = X.min() - 0.5, X.max() + 0.5
    y_min, y_max = y.min() - 1, y.max() + 1
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel("X", fontsize=12, weight="bold")
    ax1.set_ylabel("Y", fontsize=12, weight="bold")
    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)

    # Title with step info - using set_title instead of suptitle
    title_text = (
        "Initialize: F₀(x) = mean(y)"
        if step == 0
        else f"After Tree {step}: F_{step}(x) = F_{step - 1}(x) + α·h_{step}(x)"
    )
    ax1.set_title(title_text, fontsize=11, weight="bold", pad=8, color="#1e293b")

    # Show predictions and residuals
    pred = preds[step]

    # Prediction line (linear interpolation)
    ax1.plot(
        X,
        pred,
        "r-",
        linewidth=2.5,
        label=f"Prediction: F_{step}(x)",
        zorder=2,
        alpha=0.9,
    )

    ax1.scatter(
        X,
        pred,
        s=70,
        c="#ef4444",
        edgecolors="#991b1b",
        linewidth=1.5,
        zorder=4,
        alpha=0.9,
    )

    # Draw residuals with better visibility
    for i in range(len(X)):
        residual = y[i] - pred[i]
        color = "#10b981" if abs(residual) < 0.5 else "#8b5cf6"
        ax1.plot(
            [X[i], X[i]], [y[i], pred[i]], color=color, linewidth=2, alpha=0.7, zorder=1
        )

    # Add MSE metric
    mse = np.mean((y - pred) ** 2)
    ax1.text(
        0.02,
        0.98,
        f"MSE = {mse:.3f}",
        transform=ax1.transAxes,
        fontsize=11,
        weight="bold",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="#64748b", alpha=0.9
        ),
    )

    # Legend positioned to avoid overlap
    ax1.legend(
        loc="lower right",
        fontsize=10,
        framealpha=0.95,
        edgecolor="#64748b",
        fancybox=True,
    )

    # Right plot - Tree structure
    draw_tree(ax2, trees[step], step)


# Create figure with optimized layout to prevent overlap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
plt.subplots_adjust(bottom=0.10, top=0.93, left=0.06, right=0.96, wspace=0.25)

# Add static bottom text (won't be overwritten)
fig.text(
    0.5,
    0.02,
    f"Learning rate α = {learning_rate}  |  Max depth = 2  |  Using sklearn DecisionTreeRegressor",
    ha="center",
    fontsize=10,
    color="#475569",
    style="italic",
)

# Create animation with more frames for smoother playback
frames = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]  # Hold each frame
anim = FuncAnimation(fig, animate, frames=frames, interval=850, repeat=True)

# Save as GIF with higher quality and infinite loop
writer = PillowWriter(fps=2.0)
anim.save(output, writer=writer, dpi=120)
print(f"✓ Animation saved as '{output}' (loops infinitely)")

# Optional: display the animation (comment out if just saving)
plt.show()

# ============================================================================
# Create a separate plot showing the complete gradient boosted ensemble
# ============================================================================


def draw_tree_horizontal(ax, tree_data, x_pos, tree_num):
    """Draw a single tree horizontally in black and white."""
    if tree_data is None:
        # Initial prediction (F₀) - wider box
        box = mpatches.Rectangle(
            (x_pos - 0.5, -0.2),
            1.0,
            0.4,
            edgecolor="black",
            facecolor="white",
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(
            x_pos,
            0,
            f"F₀\n{initial_pred:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
        )
        return

    # Root node - wider box
    root_y = 0.5
    box = mpatches.Rectangle(
        (x_pos - 0.55, root_y - 0.15),
        1.1,
        0.3,
        edgecolor="black",
        facecolor="white",
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(
        x_pos,
        root_y,
        tree_data["split"],
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
    )

    # Level 1 nodes
    left_node = tree_data["left"]
    right_node = tree_data["right"]

    def draw_node(node, x, y):
        if node["type"] == "leaf":
            # Leaf nodes - wider circles
            circle = mpatches.Circle(
                (x, y), 0.18, edgecolor="black", facecolor="lightgray", linewidth=1.5
            )
            ax.add_patch(circle)
            ax.text(
                x,
                y,
                f"{node['value']:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                weight="bold",
            )
        else:
            # Split nodes - wider rectangles
            box = mpatches.Rectangle(
                (x - 0.42, y - 0.12),
                0.84,
                0.24,
                edgecolor="black",
                facecolor="white",
                linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(
                x, y, node["split"], ha="center", va="center", fontsize=7, weight="bold"
            )
        return node

    # Left branch - maximum spacing
    left_x, left_y = x_pos - 0.8, 0
    ax.plot(
        [x_pos, left_x],
        [
            root_y - 0.15,
            left_y + 0.18 if left_node["type"] == "leaf" else left_y + 0.12,
        ],
        "k-",
        linewidth=1.2,
    )
    draw_node(left_node, left_x, left_y)

    # Right branch - maximum spacing
    right_x, right_y = x_pos + 0.8, 0
    ax.plot(
        [x_pos, right_x],
        [
            root_y - 0.15,
            right_y + 0.18 if right_node["type"] == "leaf" else right_y + 0.12,
        ],
        "k-",
        linewidth=1.2,
    )
    draw_node(right_node, right_x, right_y)

    # Level 2 - maximum spacing to prevent any overlap
    if left_node["type"] == "split":
        ll_x, ll_y = left_x - 0.65, -0.5
        lr_x, lr_y = left_x + 0.65, -0.5
        ax.plot([left_x, ll_x], [left_y - 0.12, ll_y + 0.18], "k-", linewidth=1)
        ax.plot([left_x, lr_x], [left_y - 0.12, lr_y + 0.18], "k-", linewidth=1)
        draw_node(left_node["left"], ll_x, ll_y)
        draw_node(left_node["right"], lr_x, lr_y)

    if right_node["type"] == "split":
        rl_x, rl_y = right_x - 0.65, -0.5
        rr_x, rr_y = right_x + 0.65, -0.5
        ax.plot([right_x, rl_x], [right_y - 0.12, rl_y + 0.18], "k-", linewidth=1)
        ax.plot([right_x, rr_x], [right_y - 0.12, rr_y + 0.18], "k-", linewidth=1)
        draw_node(right_node["left"], rl_x, rl_y)
        draw_node(right_node["right"], rr_x, rr_y)

    # Tree label below
    ax.text(
        x_pos,
        -0.9,
        f"h_{tree_num}",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        style="italic",
    )


# Create the full ensemble visualization - horizontal layout
fig_ensemble = plt.figure(figsize=(18, 6))
ax_main = fig_ensemble.add_subplot(111)

# Title
fig_ensemble.suptitle(
    "Gradient Boosted Ensemble: Sequential Tree Building",
    fontsize=16,
    weight="bold",
    y=0.98,
)

# Setup axis - much wider for generous spacing
ax_main.set_xlim(-2, 28)
ax_main.set_ylim(-1.2, 1.2)
ax_main.axis("off")

# Calculate MSE for each step
mse_values = [np.mean((y - pred) ** 2) for pred in preds]

# Draw trees from left to right with 6.5 unit spacing for generous separation
x_positions = [0, 6.5, 13.0, 19.5, 26.0]
for i, (tree_data, x_pos) in enumerate(zip(trees, x_positions)):
    draw_tree_horizontal(ax_main, tree_data, x_pos, i)

    # Add arrows with MSE reduction between trees
    if i < len(trees) - 1:
        arrow_x = x_pos + 2.3
        ax_main.annotate(
            "",
            xy=(x_positions[i + 1] - 2.3, 0),
            xytext=(arrow_x, 0),
            arrowprops=dict(arrowstyle="->", lw=2, color="black"),
        )

        # MSE reduction
        mse_reduction = mse_values[i] - mse_values[i + 1]
        arrow_center = (arrow_x + x_positions[i + 1] - 2.0) / 2
        ax_main.text(
            arrow_center,
            -0.15,
            f"MSE: {mse_values[i]:.3f}→{mse_values[i + 1]:.3f}",
            ha="center",
            va="top",
            fontsize=8,
            style="italic",
        )
        ax_main.text(
            arrow_center,
            0.15,
            f"Δ={mse_reduction:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )

# Add formula at bottom
formula_text = (
    f"F(x) = F₀ + α·h₁(x) + α·h₂(x) + α·h₃(x) + α·h₄(x)     where α = {learning_rate}"
)
ax_main.text(
    0.5,
    0.05,
    formula_text,
    transform=ax_main.transAxes,
    fontsize=11,
    ha="center",
    weight="bold",
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", linewidth=1.5
    ),
)

# Save the ensemble plot
ensemble_output = os.path.join(save_dir, "gradient_boosting_ensemble.png")
plt.tight_layout()
fig_ensemble.savefig(ensemble_output, dpi=150, bbox_inches="tight", facecolor="white")
print(f"✓ Ensemble visualization saved as '{ensemble_output}'")

plt.show()
