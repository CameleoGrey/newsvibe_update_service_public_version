
from pprint import pprint
import numpy as np
from tqdm import tqdm


class SummaryMaker():

    def __init__(self, preprocessor, vectorizer):

        self.preprocessor = preprocessor
        self.vectorizer = vectorizer

        pass

    def make_summaries(self, x):

        summaries = []
        for i in tqdm(range(len(x)), 'Making summary'):
            sentences = x[i].split('.')
            sentences = self.preprocessor.prerproc_names(sentences, remove_stub_strings=True, verbose=False)

            if len(sentences) == 0:
                print('No summary at {}'.format(i))
                summaries.append('No summary')
                continue

            sentence_vecs = self.vectorizer.vectorize_docs(
                sentences, verbose=False)

            if len(sentence_vecs) == 0:
                summaries.append('no summary')
                print('No summary news ({}): {}'.format(i, x[i]))
                continue

            sentence_vecs = np.array(sentence_vecs)
            center_vec = np.mean(sentence_vecs, axis=0)
            sentence_vecs = sentence_vecs[:4]  # taking first N sentences only
            diffs = sentence_vecs - center_vec
            distances = np.sqrt(np.linalg.norm(diffs, axis=1))
            summary = sentences[np.argmin(distances)]
            summary = summary[0].upper() + summary[1:] + '.'
            summaries.append(summary)

        return summaries
