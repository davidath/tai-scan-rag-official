import os
import sys
sys.path.append('RAGs/')  # nopep8
from naive_RAG import NaiveRAG
import utils
import pickle
import numpy as np


def load_embeddings(conf, rag_model, dataset, query):
    emb_dir = conf['Experiment']['embeddings']
    emb_file = conf['Experiment']['emb_file']
    if os.path.isdir(emb_dir):
        if os.path.isfile(emb_dir + '/' + emb_file):
            embeddings = utils.load_embeddings(sys.argv[1])
        else:
            embeddings = rag_model.make_embeddings(dataset)
    else:
        os.makedirs(emb_dir)
        embeddings = rag_model.make_embeddings(dataset)

    if conf['RAG']['retrieval_framework'] == 'fastembed':
        query_embedding = rag_model.fast_embeddings(query)
    elif conf['RAG']['retrieval_framework'] == 'huggingface':
        query_embedding = rag_model.hf_embeddings(query)
    else:
        query_embedding = rag_model.ollama_embeddings(query)
    return embeddings, query_embedding


def retrieve_relevant_ai_act(conf, rag_model, dataset, embeddings, query_embedding):
    relevant_parts = []

    kbest_arts = rag_model.annoy(embeddings['articles'],
                                 query_embedding, conf['RAG']['k_best_articles'],
                                 )
    # kbest_arts = kbest_arts[0]

    for i in range(conf['RAG']['k_best_articles']):
        relevant_parts.append(dataset['articles'][kbest_arts[i]])

    # kbest_recs = NRAG.KNN(embeddings['recitals'],
    #                       query_embedding, conf['RAG']['k_best_recitals'],
    #                       metric='euclidean')
    # kbest_recs = kbest_recs[0]

    #     for i in range(conf['RAG']['k_best_recitals']):
    #         relevant_parts.append(dataset['recitals'][kbest_recs[i]])

    # kbest_annex = NRAG.KNN(embeddings['annexes'],
    #                        query_embedding, conf['RAG']['k_best_annexes'],
    #                        metric='euclidean')
    # kbest_annex = kbest_annex[0]

    #     for i in range(conf['RAG']['k_best_annexes']):
    #         relevant_parts.append(dataset['annexes'][kbest_annex[i]])
    return relevant_parts


def run(validate=False):
    conf = utils.load_config(sys.argv[1])
    dataset = utils.load_data(sys.argv[1])

    NRAG = NaiveRAG(sys.argv[1])
    query = NRAG.query_adjust_frontend()
    print(query)

    embeddings, query_embedding = load_embeddings(conf, NRAG, dataset, query)

    if 'ai_act' in conf['Experiment']['dataset']:
        relevant_parts = retrieve_relevant_ai_act(conf, NRAG, dataset,
                                                  embeddings, query_embedding)

    output = ''.join([bs+'\n'+'\n' for bs in relevant_parts])
    output += '\n'
    prompt = f"""
        Context:<<<{''.join([i+'\n\n' for i in relevant_parts])}>>>
        Task: Answer the question using the given context above. The question is: {query}
        Output: Provide a single word representing the level of the system. Choose from [Prohibited, High risk, Medium risk, Low risk]. Do not provide any additional text or explanations.
        """
    output += NRAG.generate(prompt)
    print(output)
    output += NRAG.generate_long(query, relevant_parts)
    print(output)

    utils.export_results(sys.argv[1], output)

    # if validate:
    # labels = pickle.load(
    #     open('datasets/ai_act/ai_act_articles_tokenized_labels.pkl', 'rb'))
    # labels = [i+1 for i in range(len(embeddings['articles']))]
    # labels.append(300)
    # print(best_sentences)
    #     kgroup = [c for c, i in enumerate(labels) if i == 5]
    # data = [embeddings['articles'], query_embedding]
    # data = np.concatenate(data)
    #     labs = np.zeros(len(data), dtype=int)
    #     labs[kbest_arts] = 1
    #     labs[-1] = 2
    # labels = np.array(labels)
    # utils.plot_embeddings(data, labels)


if __name__ == "__main__":
    run(validate=True)
