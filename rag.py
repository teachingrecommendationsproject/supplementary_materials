from importlib import resources
import json
import os
import re
from sentence_transformers import util
from sentence_transformers import CrossEncoder
import pymupdf



def retrieve_candidates(encoder, desc, knowledgebase_embeddings, pool_size=7):
    try:
        query_embedding = encoder.encode(desc)
    except Exception as e:
        raise RuntimeError(f"Error al encodear la descripción '{desc[:50]}...': {e}")
 
    try:
        sim = encoder.similarity(query_embedding, knowledgebase_embeddings)
        s = sim.tolist()[0]
    except Exception as e:
        raise RuntimeError(f"Error al calcular similitud contra la knowledge base: {e}")
 
    if not s:
        raise ValueError("La knowledge base está vacía o no tiene embeddings.")
 
    pool_size = min(pool_size, len(s))
    doc_indices = [
        idx for idx, _ in sorted(enumerate(s), key=lambda x: x[1], reverse=True)[:pool_size]
    ]
    return doc_indices
 
def rerank_candidates(reranker, desc, candidate_texts, top_k=7):
    if not candidate_texts:
        return []
 
    try:
        pairs = [[desc, doc] for doc in candidate_texts]
        scores = reranker.predict(pairs)
    except Exception as e:
        print(f"  [WARN] Falló el reranking ({e}). "
              f"Se usan los primeros {top_k} candidatos del retrieval sin reordenar.")
        return candidate_texts[:top_k]
 
    ranked = sorted(zip(candidate_texts, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

def get_rag_sources(desc, encoder, knowledgebase_embeddings, knowledge_base, reranker, agent=False, client = None, model = None):
    try:
        doc_indices = retrieve_candidates(
            encoder, desc, knowledgebase_embeddings, pool_size=7
        )
        candidate_texts = [knowledge_base["page_content"][idx] for idx in doc_indices]
        sources = rerank_candidates(reranker, desc, candidate_texts, top_k=7)

        sources_analysis = []

        if agent == True:
            for src in sources:
                pmt = f"""
                The following text is an excerpt taken from a bibliographic source, meaning any descriptive data it contains is not necessarily transferable to the context, although the theoretical perspective or the conclusions derived from the study are.

                Your task is to conduct a critical examination to determine whether there are elements in the text that can be directly transferred to address the pedagogical problems identified in the comments of a teaching evaluation survey.

                The identified pedagogical problems are provided below:
                {desc}

                Excerpt from the bibliographic source:
                {src}

                Respond using the following JSON schema:
                {{"explanation" : "what can be extracted to address pedagogical problem",
                "helpfull" : "<True-or-False>"}}
                """
                print(model)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": pmt
                            }
                        ],
                    model=model,
                    max_completion_tokens=10000,
                    temperature=0,
                    )
    
                response_content = chat_completion.choices[0].message.content
                
                clean_content = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL)
                if "</think>" in clean_content:
                    clean_content = clean_content.split("</think>")[-1]
                
                json_match = re.search(r'\{.*\}', clean_content, flags=re.DOTALL)
                
                try:
                    document_analysis = json.loads(json_match.group(0)) if json_match else {"helpfull": False}
                except json.JSONDecodeError:
                    document_analysis = {"helpfull": False}

                is_helpful = document_analysis.get("helpfull", False)
                if is_helpful not in (False, "False", "false"):
                    explanation = document_analysis.get("explanation", "")
                    src = f'"{src}"\nThis source could be helpful because {explanation}\n\n'
                    print(f"\nThis source could be helpful because {explanation}\n\n")
                    sources_analysis.append(src)

            sources = sources_analysis
        
        return "\n\n".join(sources)
    except Exception as e:
        print(f"  [ERROR] Falló el RAG '{str(desc)[:60]}...': {e}")
        print("  Se continúa sin fuentes para este ítem.")
        return ""




def semantic_chunking(path, encoder, SIM_THRESHOLD=0.85):

    pdfiles = os.listdir(path)
    source = []
    page_content = []

    for pdf in pdfiles:
        if not pdf.lower().endswith(".pdf"):
            continue
            
        doc = pymupdf.open(os.path.join(path, pdf))
        full_text = ""
        num_pages = len(doc)
        
        for page_num, page in enumerate(doc):
            raw_text = page.get_text()
            
            if page_num >= num_pages * 0.5:
                match = re.search(r"^\s*(?:\d+\.?\s*|[IVX]+\.?\s*)?(?:references|referencias|bibliography|bibliografía)\s*$", raw_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    # Agregamos solo el texto antes de las referencias y cortamos
                    full_text += raw_text[:match.start()] + "\n"
                    break
                    
            full_text += raw_text + "\n"
            
        full_text = re.sub(r"(\w)-\n(\w)", r"\1\2", full_text)
        full_text = re.sub(r"(?im)^.*downloaded (by|on|from).*$", "", full_text)
        
        paragraphs = re.split(r"\n\s*\n", full_text)

        chnk = ""
        chnk_embd = None

        for prg_idx, paragraph in enumerate(paragraphs):
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if not paragraph:
                continue

            prg_embds = encoder.encode(paragraph)

            if chnk == "":
                chnk = paragraph
                chnk_embd = prg_embds
            elif util.cos_sim(chnk_embd, prg_embds).item() > SIM_THRESHOLD:
                chnk += " " + paragraph
                chnk_embd = encoder.encode(chnk)
            else:
                page_content.append(chnk)
                source.append(pdf)
                chnk = paragraph
                chnk_embd = prg_embds

        if chnk:
            page_content.append(chnk)
            source.append(pdf)


    knowledge_base = {
        "source" : source,
        "page_content" : page_content
    }

    knowledgebase_embeddings = encoder.encode(page_content)

    return knowledge_base, knowledgebase_embeddings



