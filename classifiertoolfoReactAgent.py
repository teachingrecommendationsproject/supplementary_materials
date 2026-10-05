import json
import os
from groq import Groq
import pandas as pd
import time
import re



with open("/home/gabriel/Doctorado/short_instrument_for_teaching_practices.json", "r") as f:
   instrument = json.load(f)

schemas_tp = []

for dim in instrument['Dimensions']:
    for latent_trait in dim['latent_traits']:
        latent_trait['practices']
        
        for p in latent_trait['practices']:
            dct = {
                'practice' : p['practice'],
                'description' : p['description'],
                'step_by_step_analysis' : '',
                'quote' : ''  
            }
            schemas_tp.append(dct)

path = "/home/gabriel/Doctorado/proyecto anotado de comentarios/Experimento 2 - Few shot/"            
with open(path + "english_shots.json", "r") as f:
    eng_shots = json.load(f)


 
for p in schemas_tp:
    examples = []
    for s in eng_shots:
        if p['practice'] == s['label']:
            d = dict({
                "text" : s['text'],
                "chain-of-thought" : s['chain-of-thought']
            })
            examples.append(d)
    p['examples'] = examples

schemas_tp = schemas_tp[1:]




for p in schemas_tp:
    examples = []
    for s in eng_shots:
        if p['practice'] == s['label']:
            d = dict({
                "text" : s['text'],
                "chain-of-thought" : s['chain-of-thought']
            })
            examples.append(d)
    p['examples'] = examples

schemas_tp = schemas_tp[1:]





##########################################
#
#               Classifier
#
#########################################



pmt_ef_1 ="""You are an expert in university teaching.

Help me label student comments from a teaching evaluation survey. Your task is to analyze the presence or absence of a teaching practice in the student comment. To achieve that, create a step-by-step analysis to determine whether the practice is present or absent in the text.

If you conclude that the practice is present, extract the exact quote that supports your decision. If there are no quotes about the topic, leave the field as null. Do not add any comments indicating that there are no quotes—just leave it as null.

Here is the comment of the stuent to analyze:
'"""

# comment

pmt_ef_2_lab ="""'

The practice that you have to label if it is present or absent is: '"""

# label

pmt_ef_2_des = """'. Description: """ 

# description

pmt_ef_2_ex = ". And some examples of texts that an expert labeled with this teaching practice as present:\n"

pmt_ef_3_sch1 = '''

Respond with this JSON schema:
{
"practice" : "'''

pmt_ef_3_sch2 = '''",
"label" : "<present-or-absent>",
"step-by-step-analysis" : "<your-step-by-step-analysis-of-the-comment>",
"quote" : "<your-selected-evidence or null>"
}
'''


pmt_ef_4 = """
Rules:
- Return only the JSON schema, without any comments or summaries outside of it.
- Include only verbatim quotes from the text—do not modify a single word.
- If there are no direct quotes related to the topic, leave the field empty (null). Do not add comments explaining that there are no quotes.
- Do not modify the schema or add other practices. I don't care about other practices that i'm not asking.
- Do not add ```json marked. """




pmt_mf_1 ="""You are an expert in university teaching.

Your task is to analyze student comments from faculty evaluation surveys.

We want to identify textual evidence of the presence of a dysfunctional teaching practice.

Based on the textual evidence that mentions it directly or indirectly, you have to decide whether the practice is “present" or "absent” in a binary way.

Some practices are defined as a “lack of” something. In those cases, if the comment points out a problem, deficiency, or criticism related to that aspect, that counts as evidence of the dysfunctional practice.

If you don’t find any evidence, leave the "quote" field empty (fill with null value, and do not write comments such as “no evidence found”).

For each practice, follow these steps:

Carefully read the comment.

Identify words or phrases that describe behaviors, attitudes, or situations related to the practice.

Consider whether the text expresses presence, absence, criticism, or complaint related to the practice.

If you find evidence, extract an exact textual quote (the relevant phrase or part of a phrase).

If there is no evidence, fill 'quote' field with null value.

Comment to analyze:
'"""

pmt_mf_2 ="""'

If the comment does not mention anything related with the teaching practice, complete only the field corresponding to your step-by-step reasoning to conclude that it does not refer to anything in particular, and leave all other fields blank.

'"""

pmt_mf_3 ="""'

Rules:
Return only the JSON schema, with no comments or summary outside of it.
Make sure to include only exact textual quotes, without changing a single word from the text.
Remember: if there are no textual quotes that mention each topic, fill "quote" field with null value. That is, you cannot add comments like “no quote found.” Just leave it null.
Do not add ```json marked. """


    



def classifier(comment, instrument, client, model = "qwen/qwen3.6-27b"):
    responses = []

    tps = []

    for p, schema in enumerate(schemas_tp):

        if p >= 12:
            msg = f"{pmt_mf_1}{comment}{pmt_ef_2_lab}{schema['practice']}{pmt_ef_2_des}{schema['description']}{pmt_ef_2_ex}{schema['examples']}{pmt_ef_3_sch1}{schema['practice']}{pmt_ef_3_sch2}{pmt_mf_3}"
        else:
            msg = f"{pmt_ef_1}{comment}{pmt_ef_2_lab}{schema['practice']}{pmt_ef_2_des}{schema['description']}{pmt_ef_2_ex}{schema['examples']}{pmt_ef_3_sch1}{schema['practice']}{pmt_ef_3_sch2}{pmt_ef_4}"      

        jsx = False
        try_n = 0
        max_tries = 5
        
        while jsx == False and try_n < max_tries: 
            try:
                response = client.chat.completions.create(
                    model = model,
                    messages=[{"role": "user", "content": msg}],
                    max_completion_tokens=5000,
                    reasoning_effort="default",
                    stop=None
                    )
                output = response.choices[0].message.content
                output = output.split("</think>")
                output = output[1]

                json.loads(output)
                jsx = True
                
            except Exception as e:
                try_n += 1
        
        r = {
            "practice" : schema['practice'],
            "description" : schema['description'],
            "response" : output
        }

        responses.append(r)

    return responses
