import time
import re
import json
from ZeroShotRecommendations import generate_recommendation, get_syllabus_data
from rag import get_rag_sources


def build_mas_teaching_analysis_prompt(audit_type, input_type="raw"):
    prompt = "You are an expert in higher education teaching, based on Lev Vygotsky's 'Zone of Proximal Development' (ZPD) theory.\n\n"
    
    if input_type == "classified":
        prompt += "You will receive an analysis of teaching practices detected in student comments from teaching evaluation surveys.\n\n"
    else:
        prompt += "You will receive a list of student comments from teaching evaluation surveys.\n\n"
        
    prompt += "Your task is to perform a Pedagological Audit of the instructor’s professional capabilities. Your goal is to move beyond descriptive summaries and provide a high-level conceptual diagnosis divided into two specific areas:"
    
    if audit_type == "effective":
        prompt += """
        
        Your task is to analyze the Instructional Architecture & Autonomous Base:
        1. Describe the instructor’s "Teaching Style" at a conceptual level (centered on what? oriented to what?).
        2. Identify the Autonomous Base: Define the specific pedagogical challenges the instructor handles with "unconscious competence." What are the domains where they no longer require external guidance to achieve effective student learning?

        Constraint: 
        - Do not list the practices from the report. Synthesize the "mental model" the instructor uses to deliver the course.
        - Recommendations are PROHIBITED: Do not provide advice, suggestions, or solutions. Stay exclusively within the diagnostic phase.
        - Output Style: Analytical, scholarly, and diagnostic. Avoid bulleted lists of raw data. In your description, use some verbatim quotes to illustrate your analysis.
        
        """

    elif audit_type == "zpd":
        prompt += """

        Your task is to reflect on the teaching failures to simultaneously identify two key elements:
        1. The Core Barrier: The foundational problem that acts as a major barrier to student learning and triggers other secondary issues.
        2. The ZPD: From the identified failures, determine which ones the teacher could realistically resolve with appropriate scaffolding and support (institutional resources or mentorship). Focus on 'the other side of the competence frontier' and separate them from problems that are currently entirely out of their reach.

        Finally, you must balance these two aspects to select a single, focal problem that will be the target of our subsequent intervention. For your response include a conceptual description of the selected problem.

        Constraints:
        - Do not list the practices from the report. Think at a conceptual level.
        - Mentioning strengths is PROHIBITED: Do not mention, list, or praise what the teacher does well. While you should use their strengths internally to understand their current autonomous base, they must be strictly excluded from your written response.
        - Recommendations are PROHIBITED: Do not provide advice, suggestions, or solutions. Stay exclusively within the diagnostic phase.
        - Output Style: Analytical, scholarly, and diagnostic. Avoid bulleted lists of raw data. In your description, use some verbatim quotes to illustrate your analysis."""


    if input_type == "classified":
        prompt += "Report of Teaching practices detected in student comments:"
    else:
        prompt += "List of student comments of teaching evaluation surveys:"
            
    return prompt




def mas_teaching_analysis(client, model, desc, input_type="raw"):
    prompt_effective = build_mas_teaching_analysis_prompt("effective", input_type)
    prompt_zpd = build_mas_teaching_analysis_prompt("zpd", input_type)

    effective_description = generate_recommendation(
        client=client, 
        generation_model=model, 
        prompt=f"{prompt_effective}\n{desc}",
        temperature=0.0
    )
    if effective_description:
        effective_description = re.split(pattern="</think>", string=effective_description)[-1].strip()
    
    instructional_challenges = generate_recommendation(
        client=client, 
        generation_model=model, 
        prompt=f"{prompt_zpd}\n{desc}",
        temperature=0.0
    )
    if instructional_challenges:
        instructional_challenges = re.split(pattern="</think>", string=instructional_challenges)[-1].strip()

    return {
        "autonomous_base": effective_description,
        "instructional_challenges": instructional_challenges
    }







def build_mas_recommendation_prompt(analysis_desc, course_objective=None, program=None, quotes=None, strengths=None):
    base_prompt = """You are an expert in higher education teaching, based on Lev Vygotsky's 'Zone of Proximal Development' (ZPD) theory.

    You will receive a teaching analysis that breaks down: 1. Autonomous Base: What the teacher already does well in their teaching. 2. ZPD Problems: Specific areas where the teacher can improve with support
    
    """

    extras = []
    if course_objective and program:
        extras.append("the course syllabus")
    if quotes:
        extras.append("some materials from educational research and books that could help to answer")
    if strengths:
        extras.append("some strengths identified in similar courses, which could be helpful to address some problems")

    if extras:
        if len(extras) == 1:
            base_prompt += f", plus {extras[0]}."
        else:
            base_prompt += ", plus " + ", ".join(extras[:-1]) + f" and {extras[-1]}."
    else:
        base_prompt += "."

    base_prompt += """
        
    Your task is to generate one specific recommendation to address the core challenge identified, focusing on a solution the instructor can successfully implement with appropriate support.

    Rules:
    1. Competency Bridge: Your recommendations must use current skills as a foundation to solve a problem within the ZPD.
    2. Teaching Style: Your recommendations must strictly respect the teacher's teaching style (what it is centered on and its general orientation).
    3. Avoid Overload: Do not suggest changes that require a total course re-engineering or skills that the teacher has not yet demonstrated (i.e., stay within their ZPD).

    Structure for each Recommendation: 
    1. Assign a title, and then write an organic, well-developed text that includes:
    2. A brief description of which mentioned teaching failure you are addressing.
    3. A description of your proposal to address that problem.
    4. A series of concrete and actionable steps to implement these changes in the next version of the course. These must be closely aligned with both the teaching style and the specific context of the course.

    Teaching Analysis: """

    base_prompt += str(analysis_desc)

    if course_objective and program:
        base_prompt += f"""
        Course program: 
            Course objective: "{course_objective}"

            And planning: {str(program)}"""

    if quotes:
        base_prompt += f"""
        
And here is some information from research papers that could be helpful:

{quotes}"""

    if strengths:
        base_prompt += f"""

And here are some teaching strengths identified in similar courses that could help address these problems:

{strengths}"""

    return base_prompt


def multiagent_recommendation(client, model, comments_data, course_id, input_type="raw",
                             df_syllabus=None, use_rag=False, encoder=None, 
                             knowledgebase_embeddings=None, knowledge_base=None, reranker=None, sources=None):
    """
    Ejecuta el pipeline multiagente completo para un conjunto de comentarios de un curso.
    Integra la auditoría, la búsqueda RAG (opcional o mediante sources precalculados) y la generación de recomendaciones.
    """
    print(f"Iniciando auditoría pedagógica para el curso {course_id} ({input_type})...")
    audit = mas_teaching_analysis(client, model, comments_data, input_type)
    
    course_objective, program = None, None
    if df_syllabus is not None:
        course_objective, program = get_syllabus_data(df_syllabus, course_id)
        
    quotes = sources
    if quotes is None and use_rag and encoder and knowledgebase_embeddings is not None and knowledge_base is not None and reranker:
        print(f"Recuperando papers relevantes para {course_id}...")
        quotes = get_rag_sources(
            desc=audit['instructional_challenges'],
            encoder=encoder,
            knowledgebase_embeddings=knowledgebase_embeddings,
            knowledge_base=knowledge_base,
            reranker=reranker
        )
        
    print(f"Generando recomendaciones para {course_id}...")
    prompt = build_mas_recommendation_prompt(
        analysis_desc=str(audit),
        course_objective=course_objective,
        program=program,
        quotes=quotes
    )
    
    recommendation = generate_recommendation(
        client=client,
        generation_model=model,
        prompt=prompt,
        temperature=0.0
    )
    
    return recommendation

