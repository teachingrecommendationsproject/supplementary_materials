from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd

from rag import get_rag_sources 
from collaborativeTeachingfiltered import retrieve_strengths

def getAgentools(encoder, knowledgebase_embeddings, knowledge_base, reranker, df_syllabus, schemas_tp, llm_clasificador,
                data_mst, pool_size=5):

    @tool
    def ConsultPedagogicalLiterature(consulta: str) -> str:
        fuentes = get_rag_sources(
            desc=consulta, 
            encoder=encoder, 
            knowledgebase_embeddings=knowledgebase_embeddings, 
            knowledge_base=knowledge_base, 
            reranker=reranker,
            agent=False 
        )
        return fuentes if fuentes.strip() else "No se encontró literatura relevante para esta consulta."
        
    @tool
    def RequestCourseSyllabus(course_id: str) -> str:
        try:
            sc = df_syllabus[df_syllabus['course_id'] == course_id]
            if sc.empty:
                return "Programa no encontrado para este curso en la base de datos."
            
            course_objective = sc['description'].iloc[0]
            program = sc['syllabus'].iloc[0]
            return f"Objetivo del curso: {course_objective}\n\nPlanificación:\n{program}"
        except Exception as e:
            return f"Error interno al consultar el programa: {e}"
            


    @tool
    def AnalyzeTeachingPractices(comentarios_crudos: str) -> str:
        try:
            # 3. Llamamos a tu función externa 
            resultados = classifier(
                comment=comentarios_crudos, 
                instrument=instrument, 
                client=groq_client,       # Pasamos el cliente nativo que requiere tu función
                model="qwen/qwen3.6-27b"  # O el modelo que prefieras
            )
            
            # El agente necesita que la herramienta devuelva un string.
            # Convertimos la lista de diccionarios que retorna tu función a JSON (texto legible)
            return json.dumps(resultados, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return f"No se pudo clasificar el texto crudo: {e}"


    @tool
    def RetrieveStrengthsfromOtherCourses(teaching_failures: str, course_id: str, pool_size: int = 5)-> str:
        try:
            ranked = retrieve_strengths(
                encoder=encoder,
                reranker=reranker,
                data=df_strengths,
                teaching_description=teaching_description,
                course_id=course_id,
                pool_size=7
            )
            if not ranked:
                return "No se encontraron fortalezas similares en otros cursos."
            return "\n\n".join(f"{i+1}. {desc}" for i, desc in enumerate(ranked))
        except Exception as e:
            return f"Error al recuperar fortalezas colaborativas: {e}"


    
    return [ConsultPedagogicalLiterature, RequestCourseSyllabus, AnalyzeTeachingPractices, RetrieveStrengthsfromOtherCourses]
