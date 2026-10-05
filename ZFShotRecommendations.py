import pandas as pd
import time
import sys


def format_student_comments(df, course_id):
    rvs = df['reviews'][df['course_id'] == course_id].tolist()
    if len(rvs) > 1:
        desc = ""
        for i, r in enumerate(rvs):
            desc = f"\n{desc}  Comment {i+1}: {r}\n"
    else:
        desc = str(rvs)
    return desc

def get_syllabus_data(df_syllabus, course_id):
    """Extrae el objetivo del curso y el programa desde el syllabus."""
    sc = df_syllabus[df_syllabus['course_id'] == course_id]
    if sc.empty:
        return None, None
    course_objective = sc['description'].iloc[0]
    program = sc['syllabus'].iloc[0]
    return course_objective, program

def build_recommendation_prompt(desc, input_type="raw", course_objective=None, program=None, quotes=None, other_courses_strengths = None):
    base_prompt = "You are an expert in higher education teaching, based on Lev Vygotsky's 'Zone of Proximal Development' (ZPD) theory.\n\n"
    
    if input_type == "classified":
        base_prompt += "You will receive an analysis of teaching practices detected in student comments of teaching evaluation surveys"
        desc = str(desc)
        
    else:
        base_prompt += "You will receive a list of student comments of teaching evaluation surveys"

    if quotes:
        base_prompt += ", plus some materials from educational research and books that could help to answer."
    elif course_objective and program:
        base_prompt += " and the course syllabus."

    elif other_courses_strengths:
        base_prompt += ", plus some strengths identified in similar courses, which would be helpful to aboard some problems."
        
    else:
        base_prompt += "."

    base_prompt += """
    
Your task is to identify the foundational problem that acts as a major barrier to student learning and triggers other secondary issues. Then, generate one specific recommendation to address this core challenge, focusing on a solution the instructor can successfully implement with appropriate support.

Rules:
1. Competency Bridge: Your recommendations must use current skills as a foundation to solve a problem within the ZPD.
2. Teaching Style: Your recommendations must strictly respect the teacher's teaching style (what it is centered on and its general orientation).
3. Avoid Overload: Do not suggest changes that require a total course re-engineering or skills that the teacher has not yet demonstrated (i.e., stay within their ZPD).

Structure for each Recommendation: 
1. Assign a title, and then write an organic, well-developed text that includes:
2. A brief description of which mentioned teaching failure you are addressing.
3. A description of your proposal to address that problem.
4. A series of concrete and actionable steps to implement these changes in the next version of the course. These must be closely aligned with both the teaching style and the specific context of the course.
"""

    if input_type == "classified":
        base_prompt += f"\nTeaching Analysis: {desc}"
    else:
        base_prompt += f"\nStudent comments: {desc}"

    if course_objective and program:
        base_prompt += f"""
        
Course program: 
    Course objective: "{course_objective}"
    And planning: {str(program)}"""

    if quotes:
        base_prompt += f"""
        
        And here is some information from research papers that could be helpful:
        
        {str(quotes)}"""

    return base_prompt

def generate_recommendation(client, generation_model, prompt, max_retries=3, temperature=0.0, reasoning_effort="none", max_new_tokens=16000):
    last_error = None
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=generation_model,
                temperature=temperature,
                max_completion_tokens=max_new_tokens,
                reasoning_effort=reasoning_effort,
                top_p=1,
            )
            return chat_completion.choices[0].message.content
 
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            print(f"  [WARN] Falló la generación (intento {attempt + 1}/{max_retries}): "
                  f"{e}. Reintentando en {wait}s...")
            time.sleep(wait)
 
    print(f"[ERROR] La generación falló tras {max_retries} intentos: {last_error}")
    return None
