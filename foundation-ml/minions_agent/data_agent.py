import requests
import os
import sys
import subprocess
import time
from num2words import num2words
from multiprocessing import current_process
from multiprocessing.pool import ThreadPool as Pool



# --- AGENTS ---

# --- UTILS ---
def call_llm(system_prompt, user_prompt):
    full_prompt = f"System: {system_prompt}\nUser: {user_prompt}"
    payload = {
        "model": MODEL_NAME, "prompt": full_prompt, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192}
    }
    try:
        print("   (Agent is thinking...)", end="\r")
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json() #['response']
    except Exception as e:
        return f"LLM_ERROR: {str(e)}"

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


class DataAgent:
    def __init__(self, specs, ALL_DATA_PATH):
        self.money = {}
        self.agent_type = "data"
        self.specs = specs
        self.data_path = ALL_DATA_PATH

        # self.system = (
        #     f"You are an Autonomous Data Engineer. Contest: '{specs['name']}'.\n"
        #     f"ROOT SEARCH PATH: '{os.path.abspath(ALL_DATA_PATH)}'.\n"
        #     f"DATA DESCRIPTION: {specs['data_description']}.\n\n"
            
        #     "TASK: Write a Python script to Find, Inspect, and Prepare data.\n"
        #     "LOGIC FLOW:\n"
        #     "1. SEARCH: Find the subfolder inside ROOT SEARCH PATH that matches the Contest Name.\n"
        #     "2. INSPECT: List all files in that folder.\n"
        #     "3. DECIDE STRATEGY:\n"
        #     "   - CASE A: If you find separate 'train' and 'test' files -> Load them directly.\n"
        #     "   - CASE B: If you find 'train' and 'test' and 'val' -> Merge 'val' into 'train' or ignore it, then Load train/test.\n"
        #     "   - CASE C: check in folder, if only one file csv => load this as data files.\n"
        #     "4. PREPARE: Handle missing values and Encode categoricals (OneHot/Label).\n"
        #     "5. DETECT: Check label and data, label can based on data_description, rename this col into 'target' "
        #     "6. OUTPUT: Save 'processed_train.csv' and 'processed_test.csv' to the current directory.\n"
        #     "7. PRINT: 'DATA_PROCESSED: processed_train.csv processed_test.csv' at the end.\n"
        #     "OUTPUT: ONLY valid Python code."
        # )

        # Define the strict search strategy here to inject into prompts
        self.search_strategy = (
            f"ROOT SEARCH PATH: '{self.data_path}'.\n"
            "DATA RETRIEVAL RULES (STRICT):\n"
            "1. You do NOT know the filenames yet. You must write code to find them.\n"
            "2. SEARCH: Write code to `os.walk` or `glob` inside the ROOT path to find the folder matching the Contest Name.\n"
            "3. DYNAMIC LOADING (Write this EXACT logic in Python):\n"
            "   - Get list of all CSV files in that folder.\n"
            "   - IF len(csv_files) == 1:\n"
            "       Load the single file.\n"
            "       Perform `train_test_split` to create train/test dfs.\n"
            "   - ELIF len(csv_files) >= 2:\n"
            "       Identify 'train' and 'test' files by name strings.\n"
            "       Load them separately.\n"
            "4. OUTPUT: The script must save 'processed_train.csv' and 'processed_test.csv'.\n"
        )

        self.agent_profile = """You are the world's best data scientist of an automated machine learning project (AutoML) that can find the most relevant datasets,run useful preprocessing, perform suitable data augmentation, and make meaningful visulaization to comprehensively understand the data based on the user requirements. You have the following main responsibilities to complete.
        1. Retrieve a dataset from the user or search for the dataset based on the user instruction or find the subfolder inside ROOT SEARCH PATH that matches the Contest Name.
        2. Perform data preprocessing based on the user instruction or best practice based on the given tasks.
        3. Perform data augmentation as neccesary.
        4. Extract useful information and underlying characteristics of the dataset."""

    def understand_plan(self, plan):
        summary_prompt = f"""As a proficient data scientist, summarize the following plan given by the senior AutoML project manager according to the user's requirements and your expertise in data science.
        
        # User's Requirements
            Contest: '{self.specs['name']}'.\n
            ROOT SEARCH PATH: '{self.data_path}'.\n
            DATA DESCRIPTION: {self.specs['data_description']}.\n\n
        

        # Project Plan
        {plan}
        
        The summary of the plan should enable you to fulfill your responsibilities as the answers to the following questions by focusing on the data manipulation and analysis.
        1. How to retrieve or collect the dataset(s)?
        2. How to preprocess the retrieved dataset(s)?
        3. How to efficiently augment the dataset(s)?
        4. How to extract and understand the underlying characteristics of the dataset(s)?
        
        Note that you should not perform data visualization because you cannot see it. Make sure that another data scientist can exectly reproduce the results based on your summary."""

        # print("SUMMARY PROMPT \n")
        # print(summary_prompt)

        retry = 0
        while retry < 10:
            try:
                res = call_llm(self.agent_profile, summary_prompt)
                break
            except Exception as e:
                print("system", e)
                retry += 1
                continue
        
        return res['response']


    def execute_plan(self, plan, data_path, pid):
        print(self.agent_type, "I am working with the given plan!", pid)
        # print("PLAN \n")
        # print(plan)
        # data_plan = self.understand_plan(plan)
        # print("___"*69)
        # print("DATA PLAN \n")
        # print(data_plan)

        data_plan = plan

        # Check whether the given source is accessible before running the execution --> reduce FileNotFound error
        # modality-based extraction ?

        exec_prompt = f"""As a proficient data scientist, your task is to explain **detailed** steps for data manipulation and analysis parts by executing the following machine learning development plan.
        
        # MANDATORY DATA SEARCH INSTRUCTIONS
        {self.search_strategy}

        # Plan
        {data_plan}
        
        # Potential Source of Dataset
        {self.specs['data_description']}
        
        Make sure that your explanation follows these instructions:
        - All of your explanation must be self-contained without using any placeholder to ensure that other data scientists can exactly reproduce all the steps, but do not include any code.
        - Include how and where to retrieve or collect the data.
        - Include how to preprocess the data and which tools or libraries are used for the preprocessing.
        - Include how to do the data augmentation with details and names.
        - Include how to extract and understand the characteristics of the data.
        - Include reasons why each step in your explanations is essential to effectively complete the plan.        
        Note that you should not perform data visualization because you cannot see it. Make sure to focus only on the data part as it is your expertise. Do not conduct or perform anything regarding modeling or training.
        After complete the explanations, explicitly specify the (expected) outcomes and results both quantitative and qualitative of your explanations."""

        retry = 0
        while retry < 2:
            try:
                res = call_llm(self.agent_profile, exec_prompt)
                break
            except Exception as e:
                print("system", e)
                retry += 1
                continue

        # Data LLaMA summarizes the given plan for optimizing data relevant processes
        action_result = res['response']
        # Ollama doesn't have a .usage object, so we build the dict manually.
        usage_info = {
            "prompt_tokens": res.get("prompt_eval_count", 0),
            "completion_tokens": res.get("eval_count", 0),
            "total_tokens": res.get("prompt_eval_count", 0) + res.get("eval_count", 0)
        }
        
        # Save to your tracker
        self.money[f'Data_Plan_Execution_{pid}'] = usage_info
        print(self.agent_type, "I have done with my execution!", pid)
        return action_result