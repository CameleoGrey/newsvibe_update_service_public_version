
from pprint import pprint
import numpy as np
from tqdm import tqdm
from hashlib import sha256

from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration


class SummaryMakerNN():

    def __init__(self, device = "cuda"):
        
        self.device = device
        
        self.model = BartForConditionalGeneration.from_pretrained("philschmid/bart-large-cnn-samsum")
        self.tokenizer = BartTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
        
        #self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        #self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        self.model = self.model.to(self.device)
        
        pass
    
    def make_hierarchical_summaries(self, texts, topic_labels, suppression_rate=10, random_state=45, verbose=True, ready_summaries=None):
        
        print("Making common summaries")
        if ready_summaries is None:
            text_summaries = self.make_summaries(texts, verbose)
            text_summaries = np.array(text_summaries)
        else:
            text_summaries = ready_summaries
        
        print("Making topic summaries")
        level_group_hashes = []
        for group in range( len(topic_labels) ):
            current_hashes = []
            current_group = ""
            for level in range( len(topic_labels[group]) ):
                
                if level == 0:
                    current_group += str(topic_labels[group][level])
                else:
                    current_group += "_" + str(topic_labels[group][level])
                current_group_hash = current_group
                
                #current_group += str(topic_labels[group][level]) + "_"
                #current_group_hash = current_group.encode("utf-8")
                #current_group_hash = sha256( current_group_hash )
                #current_group_hash = current_group_hash.hexdigest()
                
                current_hashes.append( current_group_hash )
            level_group_hashes.append( current_hashes )
        level_group_hashes = np.array( level_group_hashes )
        
        group_summaries = {}
        for level in range( len(level_group_hashes[0]), 0, -1 ):
            level_hashes = level_group_hashes[:, level-1]
            uniq_group_hashes = np.unique( level_hashes )
            for group_hash in uniq_group_hashes:
                group_summary = text_summaries[ level_hashes == group_hash ]
                group_summary = self.make_summary_of_summaries(group_summary, suppression_rate, random_state, verbose)
                group_summaries[ group_hash ] = group_summary
                
        return text_summaries, group_summaries
    
    def make_summary_of_summaries(self, summaries, suppression_rate=10, random_state=45, verbose=True):
        
        np.random.seed( random_state )
        while len(summaries) / suppression_rate > 1:
            np.random.shuffle( summaries )
            summary_batches = np.array_split( summaries, suppression_rate )
            for i in range( len(summary_batches) ):
                summary_batches[i] = ". ".join( summary_batches[i] )
            summaries = summary_batches
            summaries = self.make_summaries(summaries, verbose)
        group_summary = ". ".join( summaries )
        group_summary = self.make_summaries([group_summary], verbose=False)[0]
        
        return group_summary
    
    # stub method for debug
    """def make_summaries(self, texts, verbose=True):
        summaries = []
        if verbose:
            progress_bar = tqdm(range(len(texts)), 'Making summaries')
        else:
            progress_bar = range(len(texts))
        for i in progress_bar:
            
            doc = texts[i]
            doc = doc.split(".")
            if len(doc) > 2:
                summary = " ".join(doc[:2])
            else:
                summary = " ".join(doc)
            summary = summary.strip()
            
            summaries.append(summary)
            
            if verbose:
                print("*******************")
                print(doc)
                print("###################")
                print(summary)
                print()
            
        return summaries"""

    def make_summaries(self, texts, verbose=True):

        summaries = []
        if verbose:
            progress_bar = tqdm(range(len(texts)), 'Making summaries')
        else:
            progress_bar = range(len(texts))
        for i in progress_bar:
            
            doc = texts[i]
            tokenized_doc = self.tokenizer(doc, truncation=True, return_tensors="pt")
            tokenized_doc = tokenized_doc.to(self.device)
            summary_ids = self.model.generate(tokenized_doc["input_ids"], num_beams=2, min_length=0, max_length=256)
            summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            summaries.append(summary)
            
            if verbose:
                print("*******************")
                print(doc)
                print("###################")
                print(summary)
                print()
            
        return summaries
