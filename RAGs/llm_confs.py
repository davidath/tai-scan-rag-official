from base_conf import BaseConf


class PlainLLM(BaseConf):
    def __init__(self, conf):
        super().__init__(conf)

    def run(self, query, task=None):
        if task == 'risk_classification':
            prompt = self.templates.rl['prompt'].substitute(query=query)
        elif task == 'relevant_resources':
            prompt = self.templates.rr['prompt'].substitute(query=query)
        elif task == 'obligation_gen':
            prompt = self.templates.og['prompt'].substitute(query=query)
        elif task is None:
            prompt = query
        else:
            raise ValueError('Invalid task name!')

        return self.text_gen.generate(prompt)

    def __str__(self):
        return (f"PlainLLM(config='{self.conf_path}', "
                f"text_gen_model_name='{self.text_gen.model_name}', \n"
                f"templates={str(self.templates)})")


class ContextLLM(BaseConf):
    def __init__(self, conf):
        super().__init__(conf)
        self.load_context(self.conf['RAG']['context'])

    def load_context(self, file_path):
        try:
            with open(file_path, 'r') as file:
                self.context = file.read()
            return True
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return False
        except Exception as e:
            print(f"Error loading context file: {e}")
            return False

    def run(self, query, task=None):
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
        return (f"ContextLLM(config='{self.conf_path}', "
                f"text_gen_model_name='{self.text_gen.model_name}', \n"
                f"templates={str(self.templates)}, "
                f"context_loaded={self.context is not None})")
