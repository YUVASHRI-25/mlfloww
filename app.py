# country_kmeans.py
import io
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------- CONFIG ----------
USERNAME = "countryuser"
PASSWORD = "123"
N_CLUSTERS = 5
DATA_PATH = "Country-data.csv"
# ----------------------------

# ---------- Load dataset ----------
df = pd.read_csv(DATA_PATH)

# Drop country column, keep numeric
country_names = df["country"]
X = df.drop(columns=["country"]).values.astype(float)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=50)
kmeans_labels = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Cluster label mapping
cluster_labels = {
    0: "Rich & Healthy",
    1: "Poor & Unhealthy",
    2: "Trade-heavy",
    3: "Emerging Middle",
    4: "Inflation/Unstable"
}

# ---------- Helper ----------
def create_plot(X_pca, labels, user_pca=None, pred_label=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.get_cmap("tab10")
    for j, u in enumerate(np.unique(labels)):
        mask = labels == u
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[cmap(j % 10)], s=35,
            label=cluster_labels[u], alpha=0.6
        )
    if user_pca is not None:
        ax.scatter(
            user_pca[0, 0], user_pca[0, 1],
            c="red", marker="X", s=200,
            label=f"Input → {pred_label}"
        )
    ax.set_title("KMeans Clusters (Country Data)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ---------- Core logic ----------
def login_action(username, password):
    if username == USERNAME and password == PASSWORD:
        return gr.update(visible=True), "✅ Welcome to Country KMeans Explorer!"
    return gr.update(visible=False), "❌ Wrong login details"

def predict_and_visualize(*features):
    try:
        arr = np.array([features], dtype=float)
        arr_scaled = scaler.transform(arr)

        # Predict with KMeans
        km_label = int(kmeans.predict(arr_scaled)[0])
        label_name = cluster_labels[km_label]

        # PCA projection
        arr_pca = pca.transform(arr_scaled)
        img = create_plot(X_pca, kmeans_labels, arr_pca, label_name)

        return label_name, img
    except Exception as e:
        return f"Error: {e}", None

# ---------- UI ----------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌍 Country Dataset — KMeans Explorer (CSV Training)")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Login")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_msg = gr.Markdown("")

        with gr.Column(scale=2):
            gr.Markdown(
                "Login first (default: `countryuser` / `123`). "
                "Then enter all country features (numeric only) to predict the cluster."
            )

    with gr.Column(visible=False) as cluster_col:
        gr.Markdown("### 🔢 Input country features")
        inputs = [gr.Number(label=col) for col in df.drop(columns=["country"]).columns]
        predict_btn = gr.Button("Predict Cluster")

        km_out = gr.Textbox(label="Cluster", interactive=False)
        plot_out = gr.Image(label="Cluster visualization (PCA)")

    login_btn.click(fn=login_action, inputs=[username, password],
                    outputs=[cluster_col, login_msg])
    predict_btn.click(fn=predict_and_visualize, inputs=inputs,
                      outputs=[km_out, plot_out])

# ---------- Run ----------
if __name__ == "__main__":
    demo.launch()

