import re
import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def main():

    embeddings_file = sys.argv[1]
    graph_name = embeddings_file.split("/")[-1]
    wv, vocabulary = load_embeddings(embeddings_file)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv[:1000,:])

    fig, ax = plt.subplots()
    fig.set_size_inches(12,10)

    points = {}
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        label = ''.join(label.split('.')[:-1]).replace('un-original-', '')
        if len(label) == 0:
            label='original'
        others = points.get(label, {})
        xs = others.get('x', []) + [x]
        ys = others.get('y', []) + [y]
        points[label] = {'x': xs, 'y': ys}

    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(points.keys())))

    plot_data = sorted(points.items(), key = lambda x: x[0])
    markers = ['8', 's', 'p', 'P', '*']
    marker = 0
    for color, (label, coords) in zip(colors, plot_data):
        if skip_point(label):
            continue
        ax.scatter(coords['x'], coords['y'], label=label, color=color, marker = markers[marker])
        marker = (marker + 1) % len(markers)

    plt.legend(loc='upper left', ncol=6)
    plt.savefig('plots/{}.pdf'.format(graph_name))

# Filter what we plot
def skip_point(name):
    if name == 'un-bter' or name == 'un-darwini':
        return True

    tokens = name.split('-')

    if len(tokens) < 3:
        return False

    size = tokens[1]
    if size in ['3', '4']:
        return False

    return True

def load_embeddings(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        next(f_in) # skip header
        vocabulary, wv = zip(*[line.strip().split(',', 1) for line in f_in])
    vocabulary = [re.sub(".el$", "", name) for name in vocabulary]
    wv = np.loadtxt(wv, delimiter=',')
    return wv, vocabulary

if __name__ == '__main__':
    main()
