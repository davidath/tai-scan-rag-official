from base_RAG import BaseRAG
import utils
import os
import pickle
from annoy import AnnoyIndex


class NaiveRAG(BaseRAG):
    def __init__(self, conf):
        super().__init__(conf)
        self._load_embeddings()

    # AI act specific
    def _make_embeddings(self, dataset):
        path_file = self.conf['Experiment']['embeddings'] + '/' + \
            self.conf['Experiment']['emb_file']

        embeddings = {}
        if 'ai_act' in self.conf['Experiment']['dataset']:
            for part in dataset:
                embeddings[part] = self.emb_gen.embed(dataset[part])
                utils.log(part)

        self.embeddings = embeddings

        # Save embeddings as dictionary
        with open(path_file, "wb") as file:
            pickle.dump(embeddings, file)

        utils.log("Embeddings saved successfully!")

    def _load_embeddings(self):
        emb_dir = self.conf['Experiment']['embeddings']
        emb_file = self.conf['Experiment']['emb_file']
        if os.path.isdir(emb_dir):
            if os.path.isfile(emb_dir + '/' + emb_file):
                self.embeddings = utils.load_embeddings(self.conf_path)
            else:
                dataset = utils.load_data(self.conf_path)
                self._make_embeddings(dataset)
        else:
            os.makedirs(emb_dir)
            dataset = utils.load_data(self.conf_path)
            self._make_embeddings(dataset)

    def _match(self, embeddings, query_embedding, k):
        # Annoy
        index = AnnoyIndex(embeddings.shape[1], 'angular')
        for i, emb in enumerate(embeddings):
            index.add_item(i, emb)
        index.build(k)
        indices = index.get_nns_by_vector(
            query_embedding.reshape(embeddings.shape[1]), k)
        return indices
        # KNN
        # from sklearn.neighbors import NearestNeighbors
        # knn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
        # knn.fit(embeddings)
        # # select indices of k nearest neighbours
        # neighbours = knn.kneighbors(query_embedding, return_distance=False)
        # return neighbours[0]

    # AI act specific
    def _retrieve_relevant_documents(self, query, part='articles'):
        query_embedding = self.emb_gen.embed(query)
        relevant_parts = []
        dataset = utils.load_data(self.conf_path)
        kbest_arts = self._match(self.embeddings[part],
                                 query_embedding,
                                 self.conf['RAG']['k_best_articles'])
        for i in range(self.conf['RAG']['k_best_articles']):
            relevant_parts.append(dataset[part][kbest_arts[i]])
        self.relevant_parts = relevant_parts
        self.context = ''.join([i+'\n\n' for i in relevant_parts])
        print(self.context)
