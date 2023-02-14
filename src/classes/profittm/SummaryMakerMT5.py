


import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SummaryMakerMT5():
    def __init__(self):
        
        self.model_name = "csebuetnlp/mT5_multilingual_XLSum"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        pass
    
    def summarize(self, text):
        
        WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
        
        input_ids = self.tokenizer(
            [WHITESPACE_HANDLER(text)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"]
        
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=84,
            no_repeat_ngram_size=2,
            num_beams=4
        )[0]
        
        summary = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        
        return summary
        
        
        