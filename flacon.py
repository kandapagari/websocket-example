# -*- coding: utf-8 -*-
import torch
import transformers
from transformers import AutoTokenizer


class Falcon(torch.nn.Module):

    def __init__(self, model_name="tiiuae/falcon-7b-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def forward(self, text):
        sequences = self.pipeline(
            text,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        results = ''
        for seq in sequences:
            results += seq['generated_text']
        return results

    def dry_run(self):
        results = self.forward("Write a poem about Valencia.")
        print(f"Result: {results}")


if __name__ == "__main__":
    model = Falcon()
    model.dry_run()
