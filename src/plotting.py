import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Plot: 2 Objs and pareto front
def plot_obj_2d(F, xlim=(0, 1), ylim=(0, 1)):
    n_obj = F.shape[1]
    if n_obj == 2:
        nds = NonDominatedSorting()
        front_idx = nds.do(F, only_non_dominated_front=True)

        pareto_F = F[front_idx]
        non_pareto_F = np.delete(F, front_idx, axis=0)

        fig = go.Figure(
            data=go.Scatter(
                x=F[:, 0],
                y=F[:, 1],
                mode='markers',
                name='Objective Values',
                marker=dict(size=6, color='#87CEEB', opacity=0.7)
            )
        )
        fig.add_trace(go.Scatter(
            x=pareto_F[:, 0],
            y=pareto_F[:, 1],
            mode='markers',
            name='Pareto Front',
            marker=dict(size=7, color='#FF7F0E', opacity=0.9, symbol='diamond')
        ))
        fig.update_layout(
            xaxis_title='f1',
            yaxis_title='f2',
            width=600,
            height=600,
            xaxis=dict(range=list(xlim)),
            yaxis=dict(range=list(ylim))
        )
        fig.show()

def evaluate_pre_real(pre, real, title=None, figsize=(7, 6), point_size=20, show_plot=True):
    pre = np.asarray(pre, dtype=float)
    real = np.asarray(real, dtype=float)

    if pre.ndim != 2 or real.ndim != 2:
        raise ValueError("pre and real must be 2D arrays.")
    if pre.shape[1] != 2 or real.shape[1] != 2:
        raise ValueError("pre and real must have shape (n, 2).")
    if pre.shape[0] != real.shape[0]:
        raise ValueError("pre and real must have the same number of rows.")

    # row-wise Euclidean distance
    distances = np.sqrt(np.sum((pre - real) ** 2, axis=1))

    max_idx = np.argmax(distances)
    min_idx = np.argmin(distances)

    result = {
        "distances": distances,
        "max_distance": distances[max_idx],
        "max_obj_point": pre[max_idx],
        "max_f_real_point": real[max_idx],
        "min_distance": distances[min_idx],
        "min_obj_point": pre[min_idx],
        "min_f_real_point": real[min_idx],
        "mean_distance": np.mean(distances)
    }

    if show_plot:
        fig, ax = plt.subplots(figsize=figsize)

        for i in range(pre.shape[0]):
            ax.annotate(
                '',
                xy=(real[i, 0], real[i, 1]),
                xytext=(pre[i, 0], pre[i, 1]),
                arrowprops=dict(
                    arrowstyle='->',
                    color='green',
                    lw=1.0,
                    alpha=0.8,
                    shrinkA=0,
                    shrinkB=0
                )
            )

        ax.scatter(
            pre[:, 0], pre[:, 1],
            color='#87CEEB',
            s=point_size,
            alpha=0.8,
            label='pre'
        )

        ax.scatter(
            real[:, 0], real[:, 1],
            color='#FF7F0E',
            s=point_size,
            alpha=0.8,
            label='real'
        )

        ax.set_xlabel('F1')
        ax.set_ylabel('F2')
        if title is not None:
            ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()

    print(f"Max:  {result['max_distance']:.2f}, sur={result['max_obj_point']}, real={result['max_f_real_point']}")
    print(f"Min:  {result['min_distance']:.2f}, sur={result['min_obj_point']}, real={result['min_f_real_point']}")
    print(f"Mean: {result['mean_distance']:.2f}")
    print("-" * 50)

    return result




    import numpy as np

def plot_z_score(y_test, pred_mean, pred_std, bins=30, eps=1e-12):
    y_test = np.asarray(y_test)
    pred_mean = np.asarray(pred_mean)
    pred_std = np.asarray(pred_std)

    z = (y_test - pred_mean) / np.maximum(pred_std, eps)

    n_obj = z.shape[1]
    fig, axes = plt.subplots(1, n_obj, figsize=(5 * n_obj, 4), sharey=True)

    if n_obj == 1:
        axes = [axes]

    for i in range(n_obj):
        axes[i].hist(z[:, i], bins=bins)
        axes[i].set_title(f"Z distribution - f{i+1}")
        axes[i].set_xlabel("z")

    axes[0].set_ylabel("count")
    plt.tight_layout()
    plt.show()
