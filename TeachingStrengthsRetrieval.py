import json
import pandas as pd
import numpy as np
import re
from IPython.display import clear_output, display
from sentence_transformers import CrossEncoder 
from sentence_transformers import SentenceTransformer
from sentence_transformers import util



def retrieve_strengths(encoder, reranker, data, teaching_description, course_id, embeddings = None, pool_size=5):

    data = data[data['course_id2'] != course_id]
    


    failures_embeddings = encoder.encode(teaching_description)
    
    s = encoder.similarity(failures_embeddings, embeddings).tolist()[0]
     
    course_indices = [
        idx for idx, _ in sorted(enumerate(s), key=lambda x: x[1], reverse=True)[:pool_size]
    ]

    candidate_strengths =  data['description'].iloc[course_indices].tolist()
    
    pairs = [[teaching_description, txt] for txt in candidate_strengths]
    scores = reranker.predict(pairs)
 
    ranked = sorted(zip(candidate_strengths, scores), key=lambda x: x[1], reverse=True)
    
    ranked = [doc for doc, _ in ranked]

    return ranked
