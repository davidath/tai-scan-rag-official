from string import Template
import utils


class AIActQPTemplates:
    def __init__(self, conf_path):
        self.templ_config = utils.load_config(conf_path)
        self._load_templates()

    def _extract_templates(self, config, task_name):
        query_template_str = config[task_name]['query']
        prompt_template_str = config[task_name]['prompt']
        return Template(query_template_str), Template(prompt_template_str)

    def _load_templates(self):
        # Risk level task
        rl_query, rl_prompt = self._extract_templates(
            self.templ_config, 'risk_level')
        self.rl = {'query': rl_query, 'prompt': rl_prompt}

        # Relevant resources task
        rr_query, rr_prompt = self._extract_templates(
            self.templ_config, 'relevant_resources')
        self.rr = {'query': rr_query, 'prompt': rr_prompt}

        # Obligation generation task
        og_query, og_prompt = self._extract_templates(
            self.templ_config, 'obligation_gen')
        self.og = {'query': og_query, 'prompt': og_prompt}

    def __str__(self):
        return (f"AIActQPTemplates(\n"
                f"    risk_level_query={self.rl['query'].template},\n"
                f"    risk_level_prompt={self.rl['prompt'].template},\n"
                f"    relevant_resources_query={self.rr['query'].template},\n"
                f"    relevant_resources_prompt={self.rr['prompt'].template}\n"
                f"    obligation_gen_query={self.og['query'].template},\n"
                f"    obligation_gen_prompt={self.og['prompt'].template}\n"
                f")")
