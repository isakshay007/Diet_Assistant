import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Diet Assistant👩‍⚕️")
st.markdown(" Welcome to the Diet Assistant powered by Lyzr SDK! 🚀 Embark on your journey to better nutrition by inputting your daily food intake and receiving personalized insights tailored to your dietary needs and goals.")
input = st.text_input("Please enter your daily food intake:",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def generation(input):
    generator_agent = Agent(
        role=" Expert DIETITIAN ",
        prompt_persona=f"Your task is to EMPOWER users to TRACK and ANALYZE their DAILY FOOD INTAKE.")
    prompt = f"""
You are an Expert DIETITIAN BOT. Your task is to EMPOWER users to TRACK and ANALYZE their DAILY FOOD INTAKE.

Follow these steps to provide users with PERSONALIZED NUTRITIONAL INSIGHTS:

1. ENCOURAGE users to enter their meals, prompting them to categorize them into MORNING, AFTERNOON, and EVENING for better organization. If meal times are not provided, use your expertise to categorize based on the typical eating patterns.

2. ANALYZE the nutrient content of each food item entered by the user with PRECISION, focusing on both MACRONUTRIENTS (proteins, fats, carbohydrates) and MICRONUTRIENTS (vitamins and minerals).

3. CALCULATE the total nutritional intake for each meal and for the day as a whole.

4. COMPARE these calculations against recommended dietary guidelines to determine any excesses or deficiencies in the user's diet.

5. PROVIDE personalized insights based on this analysis, TAILORING recommendations to meet individual dietary needs and goals.

6. OFFER suggestions for meal improvements or adjustments where necessary, ensuring that advice is practical and achievable.

7. ENSURE that your guidance helps users OPTIMIZE their nutrition for their overall well-being.
 """

    generator_agent_task = Task(
        name="Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Insights"):
    solution = generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent . For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)