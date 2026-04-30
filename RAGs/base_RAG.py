from base_conf import BaseConf
from abc import abstractmethod
from inference_handler import get_embedding_generator


class BaseRAG(BaseConf):
    def __init__(self, conf):
        super().__init__(conf)
        self.emb_gen = get_embedding_generator(self.conf)
        self.embeddings = None
        self.context = None

    @abstractmethod
    def _make_embeddings(self, dataset):
        pass

    @abstractmethod
    def _load_embeddings(self):
        pass

    @abstractmethod
    def _retrieve_relevant_documents(self, query):
        pass

    def run(self, query, task=None, part=None):
        if part is None:
            # Get context from embeddings
            self._retrieve_relevant_documents(query)
        else:
            self._retrieve_relevant_documents(query, part)

        # Prepend context to the query if it exists
        full_query = query
        if self.context:
            full_query = f"Context:<<<{self.context}>>>\n\nQuery: {query}"

        if task == 'risk_classification':
            prompt = self.templates.rl['prompt'].substitute(query=full_query)
        elif task == 'relevant_resources':
            prompt = self.templates.rr['prompt'].substitute(query=full_query)
        elif task == 'obligation_gen':
            prompt = self.templates.og['prompt'].substitute(query=full_query)
        elif task is None:
            prompt = full_query
        else:
            raise ValueError('Invalid task name!')
        return self.text_gen.generate(prompt)

    def __str__(self):
        return (f"BaseRAG(config='{self.conf_path}', "
                f"text_gen_model_name='{self.text_gen.model_name}', \n"
                f"templates={str(self.templates)}, "
                f"embeddings_loaded={self.embeddings is not None})")
