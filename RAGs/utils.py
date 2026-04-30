import sys
from datetime import datetime
import yaml
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA


# AI_ACT_RECITAL_DEFAULT_PATH = 'ai_act_recitals_tokenized.pkl'
# AI_ACT_ARTICLE_DEFAULT_PATH = 'ai_act_articles_tokenized.pkl'
# AI_ACT_ANNEX_DEFAULT_PATH = 'ai_act_annexes_tokenized.pkl'

AI_ACT_RECITAL_DEFAULT_PATH = 'ai_act_recitals.pkl'
AI_ACT_ARTICLE_DEFAULT_PATH = 'ai_act_articles.pkl'
AI_ACT_ANNEX_DEFAULT_PATH = 'ai_act_annexes.pkl'


def log(s, label='INFO'):
    sys.stdout.write(
        label + ' [' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '] ' + str(s) + '\n')
    sys.stdout.flush()


def load_config(conf_path):
    if os.path.isfile(conf_path):
        if os.path.isfile(conf_path):
            with open(conf_path, 'r') as f:
                config = yaml.safe_load(f)
    else:
        config = yaml.safe_load(conf_path)
    assert config is not None, "Failed to load configuration."
    return config


def load_query(conf_path):
    # assert os.path.isfile(conf_path), "Configuration file was not found!"
    conf = load_config(conf_path)
    query = conf['Experiment']['query']
    return query


def load_data(conf_path):
    # assert os.path.isfile(conf_path), "Configuration file was not found!"
    conf = load_config(conf_path)
    dataset = conf['Experiment']['dataset']
    if 'ai_act' in dataset:
        rec_pkl = open(dataset+'/'+AI_ACT_RECITAL_DEFAULT_PATH, 'rb')
        art_pkl = open(dataset+'/'+AI_ACT_ARTICLE_DEFAULT_PATH, 'rb')
        ann_pkl = open(dataset+'/'+AI_ACT_ANNEX_DEFAULT_PATH, 'rb')
        return {'recitals': pickle.load(rec_pkl),
                'articles': pickle.load(art_pkl),
                'annexes': pickle.load(ann_pkl)}


def load_embeddings(conf_path):
    # assert os.path.isfile(conf_path), "Configuration file was not found!"
    conf = load_config(conf_path)
    emb_dir = conf['Experiment']['embeddings']
    emb_file = conf['Experiment']['emb_file']
    embeddings = conf['Experiment']['embeddings']
    if 'ai_act' in embeddings:
        emb_pkl = open(emb_dir + '/' + emb_file, 'rb')
        return (pickle.load(emb_pkl))


def export_results(conf_path, output):
    assert os.path.isfile(conf_path), "Configuration file was not found!"
    conf = load_config(conf_path)
    res_path = conf['Experiment']['results_path']
    results_txt = conf['Experiment']['results_txt']
    if not os.path.isdir(res_path):
        os.makedirs(res_path)
    else:
        with open(res_path + '/' + results_txt, 'w') as txt_file:
            txt_file.write(output)


def plot_embeddings(data, labels=None):
    import mplcursors
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from sklearn.manifold import TSNE
    # Perform PCA to reduce to 2D
    pca = TSNE(n_components=2)
    projected = pca.fit_transform(data)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        projected[:, 0], projected[:, 1], c=labels, cmap='tab20', alpha=0.7)
    # Mark the last data point with an 'X'
    ax.scatter(projected[-1, 0], projected[-1, 1],
               color='black', marker='X', s=100, label="Last point")

    # Title and axis
    ax.set_title('PCA 2D Projection with Hover Labels')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # Add hover cursor
    cursor = mplcursors.cursor(scatter, hover=True)

    # Customize the hover annotation
    @cursor.connect("add")
    def on_hover(sel):
        index = sel.index
        if labels is not None:
            sel.annotation.set_text(f"Index: {index}\nLabel: {labels[index]}")
        else:
            sel.annotation.set_text(f"Index: {index}")

    plt.tight_layout()
    plt.show()
