from abc import ABC, abstractmethod
from inference_handler import get_text_generator
from ai_act_qp_templates import AIActQPTemplates
import utils


class BaseConf(ABC):
    def __init__(self, conf):
        self.conf_path = conf
        self.conf = utils.load_config(conf)
        self.text_gen = get_text_generator(self.conf)
        self.templates = AIActQPTemplates(
            self.conf['Experiment']['io_templates'])

    def _get_default_record_data(self):
        return {
            'role': self.conf['front-end']['role'],
            'domain': self.conf['front-end']['domain'],
            'type': self.conf['front-end']['type'],
            'input_data': self.conf['front-end']['input_data'],
            'intended_use': self.conf['front-end']['intended_use']
        }

    def make_query_from_config(self, task='risk_classification'):
        data = self._get_default_record_data()

        if task == 'risk_classification':
            template = self.templates.rl['query']
        elif task == 'relevant_resources':
            template = self.templates.rr['query']
        elif task == 'obligation_gen':
            template = self.templates.og['query']
        else:
            raise ValueError('Invalid task name!')

        return template.substitute(**data)

    def make_query_from_record(self, record, task='risk_classification'):
        if task == 'risk_classification':
            template = self.templates.rl['query']
        elif task == 'relevant_resources':
            template = self.templates.rr['query']
        elif task == 'obligation_gen':
            template = self.templates.og['query']
        else:
            raise ValueError('Invalid task name!')

        return template.substitute(**record)

    @abstractmethod
    def run(self, query, task=None):
        pass

    def __str__(self):
        return (f"PlainLLM(config='{self.conf_path}', "
                f"text_gen_model_name='{self.text_gen.model_name}', \n"
                f"templates={str(self.templates)})")
