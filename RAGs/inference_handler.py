import os
import torch
import numpy as np
from ollama import Client
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from torch.utils.data import DataLoader, TensorDataset
from openai import OpenAI


def mean_pooling(model_output, attention_mask):
    # (batch_size, seq_len, hidden_dim)
    token_embeddings = model_output.last_hidden_state
    unsqueeze = attention_mask.unsqueeze(-1)
    input_mask_expanded = unsqueeze.expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class HugFaceEmbeddingGenerator():
    def __init__(self, conf):
        self.cache_dir = conf['Experiment']['cache_dir']
        self.model_name = conf['RAG']['emb_gen']['model']
        self.batch_size = conf['RAG']['emb_gen']['hf_batch_size']
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        self.model.to('cpu')
        self.model.eval()

    def embed(self, sentences):
        # Tokenize and create dataset
        emb_array = self.tokenizer(sentences, return_tensors="pt",
                                   padding=True, truncation=True)
        dataset = TensorDataset(
            emb_array['input_ids'], emb_array['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        gather = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = [x.to("cpu") for x in batch]

                with torch.amp.autocast('cpu'):
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask)

                # Mean Pooling
                # (batch_size, seq_len, hidden_size)
                sentence_embeddings = mean_pooling(
                    outputs, attention_mask)

                # Optionally normalize embeddings
                # sentence_embeddings = torch.nn.functional.normalize(
                # sentence_embeddings, p=2, dim=1)
                gather.append(sentence_embeddings.cpu())

        final_embeddings = torch.cat(gather, dim=0)
        return final_embeddings.numpy()


class OllamaEmbeddingGenerator():
    def __init__(self, conf, host='http://localhost:11434'):
        self.model_name = conf['RAG']['emb_gen']['model']
        self.model = Client(host=host)

    def embed(self, sentences):
        emb = self.model.embed(model=self.model_name,
                               input=sentences)
        emb = np.array(emb['embeddings'])
        return emb


class HugFaceTextGenerator():
    def __init__(self, conf):
        self.cache_dir = conf['Experiment']['cache_dir']
        self.model_name = conf['RAG']['text_gen']['model']
        self.seed = int(conf['Experiment']['seed'])
        self.temp = float(conf['Experiment']['llm_temp'])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir)
        self.model.to('cpu')

    def generate(self, prompt):
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs.to('cpu')

        # Generate output
        outputs = self.model.generate(
            inputs["input_ids"],
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=self.temp
        )

        # Decode and print
        response = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        return response


class OllamaTextGenerator():
    def __init__(self, conf, host='http://localhost:11434'):
        self.model_name = conf['RAG']['text_gen']['model']
        self.model = Client(host=host)
        self.seed = int(conf['Experiment']['seed'])
        self.temp = float(conf['Experiment']['llm_temp'])

    def generate(self, prompt):
        response = self.model.generate(model=self.model_name,
                                       options={'seed': self.seed,
                                                'temperature': self.temp},
                                       prompt=prompt)
        return response.response


class FHGGatewayEmbeddingGenerator():
    def __init__(self, conf):
        self.model_name = conf['RAG']['emb_gen']['model']
        self.client = OpenAI(
            api_key="xxxx",
            default_headers={
                "Authorization": f"Basic {conf['RAG']['auth_token']}"},
            base_url=os.environ.get('FHG_GATEWAY_BASE_URL')
        )

    def embed(self, sentences):
        emb = self.client.embeddings.create(
            model=self.model_name,
            input=sentences
        )
        return np.array([i.embedding for i in emb.data])


class FHGGatewayTextGenerator():
    def __init__(self, conf):
        self.model_name = conf['RAG']['text_gen']['model']
        self.client = OpenAI(
            api_key="xxxx",
            default_headers={
                "Authorization": f"Basic {conf['RAG']['auth_token']}"},
            base_url=os.environ.get('FHG_GATEWAY_BASE_URL')
        )

    def generate(self, prompt):
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=32768
        )
        return response.choices[0].text


def get_embedding_generator(conf):
    framework = conf['RAG']['emb_gen']['framework']
    if framework == 'huggingface':
        return HugFaceEmbeddingGenerator(conf)
    elif framework == 'ollama':
        return OllamaEmbeddingGenerator(conf)
    elif framework == 'fhg':
        return FHGGatewayEmbeddingGenerator(conf)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


def get_text_generator(conf):
    framework = conf['RAG']['text_gen']['framework']
    if framework == 'huggingface':
        return HugFaceTextGenerator(conf)
    elif framework == 'ollama':
        return OllamaTextGenerator(conf)
    elif framework == 'fhg':
        return FHGGatewayTextGenerator(conf)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
