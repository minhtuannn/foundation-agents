import requests
import os
import sys
import subprocess
import time
from num2words import num2words
from multiprocessing import current_process
from multiprocessing.pool import ThreadPool as Pool
import json 

from minions_agent.data_agent import DataAgent 
from minions_agent.operation_agent import OperationAgent 

# --- MANAGER ---

agent_manager_profile = """You are an experienced senior project manager of a automated machine learning project (AutoML). You have two main responsibilities as follows.
1. Receive requirements and/or inquiries from users through a well-structured JSON object.
2. Using recent knowledge and state-of-the-art studies to devise promising high-quality plans for data scientists, machine learning research engineers, and MLOps engineers in your team to execute subsequent processes based on the user requirements you have received.
"""
json_plan = """Each of the following plans should cover the entire process of machine learning model development when applicable based on the given requirements, i.e., from problem formulation to deployment.
Please ansewer your plans in list of the JSON object with `title` and `steps` keys."""

basic_profile = """You are a helpful, respectful and honest "human" assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

plan_conditions = """
- Ensure that your plan is up-to-date with current state-of-the-art knowledge.
- Ensure that your plan is based on the requirements and objectives described in the above JSON object.
- Ensure that your plan is designed for AI agents instead of human experts. These agents are capable of conducting machine learning and artificial intelligence research.
- Ensure that your plan is self-contained with sufficient instructions to be executed by the AI agents. 
- Ensure that your plan includes all the key points and instructions (from handling data to modeling) so that the AI agents can successfully implement them. Do NOT directly write the code.
- Ensure that your plan completely include the end-to-end process of machine learning or artificial intelligence model development pipeline in detail (i.e., from data retrieval to model training and evaluation) when applicable based on the given requirements.
""" 

possible_states = {
    "INIT": "",
    "PLAN": "",
    "ACT": "",
    "PRE_EXEC": "",
    "EXEC": "",
    "POST_EXEC": "",
    "REV": "",
    "RES": "",
}


# --- UTILS ---
def call_llm(system_prompt, user_prompt):
    full_prompt = f"System: {system_prompt}\nUser: {user_prompt}"
    payload = {
        "model": MODEL_NAME, "prompt": full_prompt, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 16384}
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

class AutonomousManager:
    
    def __init__(self, specs, ALL_DATA_PATH):
        self.specs = specs
        self.n_plans = 1
        self.plans = [] 
        self.money = {}
        self.timer = {}
        self.data_path = ALL_DATA_PATH
        self.agent_type = "manager"
        self.plan_path = './plan_path'
        self.n_candidates = 3
        self.full_pipeline = True
        self.device = 0
        self.n_attempts = 0

        # self.task = task 

        self.make_plan_data = (
            f"You are an Autonomous Data Engineer. Contest: '{self.specs['name']}'.\n"
            f"ROOT SEARCH PATH: '{self.data_path}'.\n"
            f"DATA DESCRIPTION: {self.specs['data_description']}.\n\n"
            
            "TASK: Write a Python script to Find, Inspect, and Prepare data.\n"
            "LOGIC FLOW:\n"
            "1. SEARCH: Find the subfolder inside ROOT SEARCH PATH that matches the Contest Name.\n"
            "2. INSPECT: List all files in that folder.\n"
            "3. DECIDE STRATEGY:\n"
            "   - CASE A: If you find separate 'train' and 'test' files -> Load them directly.\n"
            "   - CASE B: If you find 'train' and 'test' and 'val' -> Merge 'val' into 'train' or ignore it, then Load train/test.\n"
            "   - CASE C: check in folder, if only one file csv => load this as data files.\n"
            "4. PREPARE: Handle missing values and Encode categoricals (OneHot/Label).\n"
            "5. DETECT: Check label and data, label can based on data_description, rename this col into 'target' "
            "6. OUTPUT: Save 'processed_train.csv' and 'processed_test.csv' to the current directory.\n"
            "7. PRINT: 'DATA_PROCESSED: processed_train.csv processed_test.csv' at the end.\n"
            "OUTPUT: ONLY valid Python code."
        )

    def make_plans(self, is_revision=False):
        # planning should include action_id, completion_status, action_dependencies (with required prior action ids), and
        # instruction (i.e., prompt to tell how Prompt Agent should parse user's input prompt (e.g., what keys should be included etc.)) for the repsective agent(s) responding to the given tasks

        start_time = time.time()

        """ 
        Freeze loop plan fail + retrieve knowledge + searching knowledge(in plan_prompt) -> good tool but will use later
        """
        # # retrieve relevant knowledge/expereince (from internal and external sources) for effective planning
        # if self.plan_knowledge == None and self.rap and self.inj in [None, 'pre']:
        #     self.plan_knowledge = retrieve_knowledge(self.user_requirements, self.req_summary, llm=self.llm, inj=self.inj)
        # else:
        #     self.plan_knowledge, self.post_noise = retrieve_knowledge(self.user_requirements, self.req_summary, llm=self.llm, inj=self.inj)
        #     self.plan_knowledge = f""""{self.plan_knowledge}\r\nHere is a list of knowledge written by an AI agent for a relevant task:\r\n{self.post_noise}"""

        # print_message(
        #     self.agent_type,
        #     f"Now, I am making a set of plans for you based on your requirements and the following knowledge 💭.\n{self.plan_knowledge}",
        # )
        # self.timer['retrieve_knowledge'] = time.time() - start_time
        
        # Define strict data loading rules for the Manager to include in the plan

        # f"1. The data is located in a subfolder inside: '{self.data_path}'.\n"
            # f"2. You must instruct the agent to search for the folder matching the contest name '{self.specs['name']}'.\n"
            # f"3. In the folder contest name, you must instruct the agent to search file, which include this File Loading Strategy in the plan:\n"
            # "   - CASE A: If separate 'train' and 'test' files exist -> Load them.\n"
            # "   - CASE B: If 'train', 'test', 'val' exist -> Merge 'val' into 'train'.\n"
            # "   - CASE C: If only one file exists -> Load and split (80/20)."

        data_loading_rules = """
            f"CRITICAL DATA LOADING INSTRUCTIONS:\n"

            STRICT CODING REQUIREMENTS WHEN LOADING DATA:

            1. FOLDER SEARCH LOGIC:
            - Use `os.walk` on root path: {}
            - Look specifically into the `dirs` (folder names), NOT files.
            - Find the folder that *contains* the string "Titanic". Save this as `target_folder`.

            2. FILE LOADING LOGIC (Must implement this IF/ELSE structure):
            - List all .csv files in `target_folder`.
            - IF len(csv_files) == 1:
                - Load the single file.
                - Split it 80/20 into `train_df` and `test_df`.
            - ELIF len(csv_files) >= 2:
                - Identify the file with "train" in the name -> `train_df`.
                - Identify the file with "test" in the name -> `test_df`.
            3. OUTPUT: The script must save 'train.csv' and 'test.csv'.
                """


        data_loading_rules = data_loading_rules.format(self.data_path)

        # Independent Planning (i.e., The agent does not know how it previously made the plans. Pros: Significantly less contexnt length consumption --> have room for knowledge sources, Cons: Diversity is not guaranteed.)
        plan_prompt = f"""Now, I want you to devise an end-to-end actionable plan according to the user's requirements described in the following JSON object.

        # User Requirements & Data Info
        ```json
        {{
            "Contest Name": "{self.specs['name']}",
            "Data Description": "{self.specs['data_description']}",
            "Requirements": "{self.specs['requirements']}"
        }}
        ```

        When devising a plan, follow these instructions and do not forget them:
        {plan_conditions}
        {data_loading_rules}

        """
        
        start_time = time.time()
        for i in range(1, self.n_plans + 1):
     
            while True:
                try:
                    response = call_llm(agent_manager_profile, plan_prompt)
                    break
                except Exception as e:
                    print("system", e)
                    continue
            # plan = response['message']['content'].strip()
            plan = response['response']
            self.plans.append(plan)
            # Ollama doesn't have a .usage object, so we build the dict manually.
            usage_info = {
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
            }
            
            # Save to your tracker
            self.money[f'manager_plan_{i}'] = usage_info
        self.timer['planning'] = time.time() - start_time
        return self.plans


    def execute_plan(self, plan):
        # langauge (text) based execution
        identity = current_process()._identity
        if identity:
            pid = identity[0]
        else:
            pid = 1  # Default ID for the Main Process/Thread

        start_time = time.time()
        print("PLAN from AGENT:")
        # print(plan)

        # Data Agent generates the results after execute the given plan
        data_llama = DataAgent(self.specs, self.data_path)

        data_result = data_llama.execute_plan(plan, self.data_path, pid)
        print("___"*69)
        print("DATA RESULT from AGENT:")
        # print(data_result)
        self.timer[f'data_execution_{pid}'] = time.time() - start_time
        self.money['Data'] = data_llama.money

        # # Model Agent summarizes the given plan for optimizing data relevant processes
        # # Model Agent generates the results after execute the given plan
        # start_time = time.time()
        # model_llama = ModelAgent(
        #     user_requirements=self.user_requirements,
        #     llm=self.llm,
        #     rap=self.rap,
        #     decomp=self.decomp,
        # )
        # model_result = model_llama.execute_plan(
        #     k=self.n_candidates, project_plan=plan, data_result=data_result, pid=pid
        # )
        # self.timer[f'model_execution_{pid}'] = time.time() - start_time
        # self.money['Model'] = model_llama.money
        
        return {"DATA": data_result} #, "model": model_result}


    def choose_solution(self, plans):
        print(
                    self.agent_type,
                    "With the above plan(s), our 🦙 Data Agent and 🦙 Model Agent are going to find the best solution for you!",
                )

        start_time = time.time()
        # Parallelization
        with Pool(self.n_plans) as pool:
            self.action_results = pool.map(self.execute_plan, plans)
        self.timer['plan_execution_total'] = time.time() - start_time
        print("___"*69)
        print("ACTION RESULT from AGENT:")
        # print(self.action_results)
        return self.action_results


    def verify_solution(self, solution):

        ### Remove model for now to focus on DataAgent

        identity = current_process()._identity
        if identity:
            pid = identity[0]
        else:
            pid = 1  # Default ID for the Main Process/Thread

        
        start_time = time.time()
        
        is_pass = False

        # pre-execution verification
        verification_prompt = """Given the proposed solution and user's requirements, please carefully check and verify whether the proposed solution 'pass' or 'fail' the user's requirements.
        
        **Proposed Solution and Its Implementation**
        Data Manipulation and Analysis: {}
        
        **User Requirements**
        ```json
        {}
        {}
        ```
                
        Answer only 'Pass' or 'Fail'
        """

        # print("___"*69)
        # print("INPUT to from VERIFY AGENT:")
        # print(solution["data"])
        # print("___"*69)
        # print(self.specs['data_description'])
        # print("___"*69)
        # print(self.specs['requirements'])

        prompt = verification_prompt.format(
            solution["DATA"],  self.specs['data_description'], self.specs['requirements']
        )
        

        while True:
            try:
                res = call_llm(basic_profile, prompt)
                break
            except Exception as e:
                print_message("system", e)
                continue

        ans = res['response']
        print("___"*69)
        print("VERIFY RESULT from AGENT:")
        # print(ans)
        is_pass = "pass" in ans.lower()
        self.money['manager_execution_verification'] = {
                                                        "prompt_tokens": res.get("prompt_eval_count", 0),
                                                        "completion_tokens": res.get("eval_count", 0),
                                                        "total_tokens": res.get("prompt_eval_count", 0) + res.get("eval_count", 0)
                                                        }
        
        self.timer[f'execution_verification_{pid}'] = time.time() - start_time

        return is_pass

    def implement_solution(self, selected_solution):
        # with open(f"prompt_pool/{self.task}.py") as file:
        #     template_code = file.read()        
        # code-based execution
        uid = 0
        self.code_path = f"/{uid}_p{self.n_plans}_{'full' if self.full_pipeline else ''}"
        ops_llama = OperationAgent(
            specs=self.specs,
            code_path=self.code_path,
            device=self.device,
            data_path = self.data_path
        )
        ops_result = ops_llama.implement_solution(
            code_instructions=selected_solution, 
            full_pipeline=self.full_pipeline, 
            # code=template_code
        )
        self.money['Operation'] = ops_llama.money
        return ops_result

    
    def generate_code(self, action_results):

        """
        Purpose: one
        """
        # Code Execution stage
        start_time = time.time()
        
        data_plan_for_execution = ""
        model_plan_for_execution = ""
        for action in action_results:
            if action["pass"]:
                data_plan_for_execution = (
                    data_plan_for_execution + action["DATA"] + "\n"
                )
                # model_plan_for_execution = (
                #     model_plan_for_execution + action["model"] + "\n"
                # )

        # Summarize the passed plan for operation llama to write and execute the code
        upload_path = (
            f"This is the retrievable data path: {self.data_path}."
            if self.data_path
            else ""
        )
        summary_prompt = f"""As the project manager, please carefully read and understand the following instructions suggested by data scientists and machine learning engineers. Then, select the best solution for the given user's requirements.
        
        - Instructions from Data Scientists
        {data_plan_for_execution}
        If there is no predefined data split or the data scientists suggest the data split other than train 70%, validation 20%, and test 10%, please use 70%, 20%, and 10% instead for consistency across different tasks. {upload_path}
        You should exclude every suggestion related to data visualization as you will be unable to see it.
        
        # - Instructions from Machine Learning Engineers
        # model_plan_for_execution                    
        
        - User's Requirements
        {self.specs['requirements']}
        {self.specs['data_description']}
        
        Note that you must select only ONE promising solution (i.e., one data processing pipeline and one model from the top-{num2words(self.n_candidates)} models) based on the above suggestions.
        After choosing the best solution, give detailed instructions and guidelines for MLOps engineers who will write the code based on your instructions. Do not write the code by yourself. Since PyTorch is preferred for implementing deep learning and neural networks models, please guide the MLOPs engineers accordingly.
        Make sure your instructions are sufficient with all essential information (e.g., complete path for dataset source and model location) for any MLOps or ML engineers to enable them to write the codes using existing libraries and frameworks correctly."""
        
        print("SUMMARY_PROMPT_GENCODE \n")
        print(summary_prompt)
        self.implementation_result = self.implement_solution(data_plan_for_execution)
        print('system', f'{self.code_path}, <<< END CODING, TIME USED: {time.time() - start_time} SECS >>>')
        self.timer['implementation'] = time.time() - start_time
        
        return self.implementation_result

        
    # end def
    def run_phase(self, phase_name, agent, filename, max_retries=3):
        print(f"\n🔵 PHASE: {phase_name}")
        feedback = ""
        
        for attempt in range(max_retries):
            print(f"   Attempt {attempt+1}...")
            raw_response = agent.generate(feedback)
            filepath, _ = save_code(raw_response, filename)
            
            result = execute_script(filepath)
            
            if result and result.returncode == 0:
                print(f"   ✅ Success.")
                return True, result.stdout
            else:
                print(f"   ❌ Failed.")
                err = result.stderr.strip() if result else "Timeout"
                print(f"      Err: {err[-200:]}")
                feedback = f"Runtime Error: {err}"
        
        return False, "Failed"

    def start(self):
        print(f"STARTING JOB: {self.specs['name']}")
        
        # Phase 1: Smart Data Search
        print("Planning...")
        plans = self.make_plans()

        print(plans)

        # print("Choosing...")        
        # action_results = self.choose_solution(self.plans)
        # print(action_results)
        # print("Verifying...")
        # verify = self.verify_solution(action_results[0]) 
        
        # print(
        #                 self.agent_type,
        #                 "I am now verifying the solutions found by our Agent team 🦙.",
        #             )
                    
        # start_time = time.time()
        # # Parallelization
        # with Pool(self.n_plans) as pool:
        #     verification_result = pool.map(self.verify_solution, action_results)
        # self.timer['execution_verification_total'] = time.time() - start_time

        # for i, result in enumerate(verification_result):
        #     action_results[i]["pass"] = result

        # # """  
        # # This code for save result about each plan, run after choose which one is the best plan
        # # """
        # os.makedirs(self.plan_path, exist_ok=True)
        # for i, action in enumerate(action_results):
        #     if action["pass"]:
        #         # save pass plan
        #         filename = f"{self.plan_path}/plan_{i}.json"
        #         os.makedirs(os.path.dirname(filename), exist_ok=True)
        #         with open(filename, "w") as f:
        #             json.dump(action, f)
        #             print(
        #                 self.agent_type,
        #                 f"Saved a pass plan: {self.plan_path}/plan_{i}.json",
        #             )
        # print(self.agent_type, "let our Operation Agent 🦙 implement and evaluate these solutions 👨🏻‍💻")

        # print("Generating code...")
        # # print(action)

      
        # for action in self.action_results:
        #         action["pass"] = True
        # gen_code = self.generate_code(action_results=action_results)
        # print(gen_code)
        



