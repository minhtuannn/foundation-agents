import requests
import os
import sys
import subprocess
import time
from num2words import num2words
from multiprocessing import current_process
from multiprocessing.pool import ThreadPool as Pool
import selectors


agent_profile = """You are the world's best MLOps engineer of an automated machine learning project (AutoML) that can implement the optimal solution for production-level deployment, given any datasets and models. You have the following main responsibilities to complete.
1. Write accurate Python codes to retrieve/load the given dataset from the corresponding source.
2. Write effective Python codes to preprocess the retrieved dataset and saved dataset.
3. Write precise Python codes to retrieve/load the given model and optimize it with the suggested hyperparameters.
4. Write efficient Python codes to train/finetune the retrieved model.
5. Write suitable Python codes to prepare the trained model for deployment. This step may include model compression and conversion according to the target inference platform.
6. Write Python codes to build the web application demo using the Gradio library.
7. Run the model evaluation using the given Python functions and summarize the results for validation againts the user's requirements.
"""

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

def execute_script(script_name, work_dir = ".", device="0"):    
    if not os.path.exists(os.path.join(work_dir, script_name)):
        raise Exception(f"The file {script_name} does not exist.")
    try:
        script_path = script_name
        device = device        
        cmd = f"CUDA_VISIBLE_DEVICES={device} python -u {script_path}"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=work_dir)

        stdout_lines = []
        stderr_lines = []

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)

        while process.poll() is None and selector.get_map():
            events = selector.select(timeout=1)

            for key, _ in events:
                line = key.fileobj.readline()
                if key.fileobj == process.stdout:
                    # print("STDOUT:", line, end =" ")
                    stdout_lines.append(line)
                else:
                    # print("STDERR:", line, end =" ")
                    stderr_lines.append(line)

        for line in process.stdout:
            line = line
            # print("STDOUT:", line, end =" ")
            stdout_lines.append(line)
        for line in process.stderr:
            line = line
            # print("STDERR:", line, end =" ")
            stderr_lines.append(line)

        return_code = process.returncode

        if return_code != 0:
            observation = "".join(stderr_lines)
        else:
            observation = "".join(stdout_lines)
        if observation == "" and return_code == 0:
            # printed to stderr only
            observation = "".join(stderr_lines)
        return return_code, "The script has been executed. Here is the output:\n" + observation
    
    except Exception as e:
        print("++++", "Wrong!")
        # raise Exception(f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed.")
        return -1, f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed."


class OperationAgent:
    def __init__(self, specs, code_path, device=0, data_path=None ):
        # setup Farm Manager
        self.agent_type = "operation"
        # self.llm = llm
        # self.model = AVAILABLE_LLMs[llm]["model"]
        self.experiment_logs = []
        self.specs = specs
        self.root_path = './agent_workspace_minions' #"./" # + f"{code_path}"
        self.code_path = code_path
        self.device = device
        self.money = {}
        self.data_path = data_path

    def self_validation(self, filename):
        rcode, log = execute_script(filename, device=self.device)
        return rcode, log

    def get_fix_advice(self, log):
        advice = ""
        
        # Map error patterns to specific advice
        common_errors = {
            "IndexError": "Check your list/array lengths before accessing indices (e.g., `if len(x) > 1:`). Do not assume fixed sizes.",
            "KeyError": "You are trying to access a dictionary key or DataFrame column that does not exist. Print `df.columns` or `dict.keys()` to verify names first.",
            "ModuleNotFoundError": "You are trying to import a library that is not installed. Switch to standard libraries (like `os`, `json`, `random`) or `scikit-learn` / `pandas`.",
            "FileNotFoundError": "The file path is wrong. Use `os.listdir()` to see what files are actually there before opening them.",
            "SyntaxError": "You have a typo in your code (missing parenthesis, colon, or indentation). Check the line number in the traceback.",
            "NameError": "You are using a variable that hasn't been defined yet. Make sure you define it before using it.",
            "ValueError": "There is a data type mismatch or invalid value. Check the inputs to your function."
        }

        # Iterate and find matches
        for error_name, specific_tip in common_errors.items():
            if error_name in log:
                advice += f"\n- **{error_name} Detected**: {specific_tip}"
                
        # Default fallback if no specific error found
        if not advice:
            advice = "\n- **Unknown Error**: Analyze the Traceback below carefully. Add `print()` statements to debug values before the crash."
            
        return advice

    def implement_solution(self, code_instructions, full_pipeline=True, code="", n_attempts=5):
        # TODO: Lock template code for reduce complex workflow

        # print(
        #     self.agent_type,
        #     f"I am implementing the following instruction:\n\,r{code_instructions}",
        # )


        print(
            self.agent_type,
            "I am implementing the following instruction"
        )


        log = "Nothing. This is your first attempt."
        error_logs = []
        code = code  # if a template/skeleton code is provided
        iteration = 0
        completion = None
        action_result = ""
        rcode = -1


        while iteration < n_attempts:
            try:
                dynamic_advice = ""
                if "Traceback" in log or "Error" in log:
                    dynamic_advice = self.get_fix_advice(log) 
                    # Wrapper specifically for the prompt
                    fix_section = f"""
                    # CRITICAL FIX INSTRUCTIONS
                    The previous attempt failed. You MUST follow this advice:
                    {dynamic_advice}
                    """
                else:
                    # If first attempt, this section should be EMPTY
                    fix_section = ""


                exec_prompt = """Carefully read the following instructions to write Python code for {} task.
                {}
                
                # Previously Written Code
                ```python
                {}
                ```
                
                # Error from the Previously Written Code
                {}
                

                Note that you need to write the python code for the {}. If saving model is required, you must save the trained model to "./agent_workspace/trained_models" directory.
                Start the python code with "```python". Please ensure the completeness of the code so that it can be run without additional modifications.
                If there is any error from the previous attempt, please carefully fix it first."""
                pipeline = (
                    "entire machine learning pipeline (from data retrieval to model deployment via Gradio)"
                    if full_pipeline
                    else "modeling pipeline (from data retrieval to model saving)"
                )
                exec_prompt = exec_prompt.format(
                    self.specs['requirements'],
                    code_instructions,
                    code,
                    log,
                    "data processing part",
                )
                ##############

                # system = """
                #         You are a Data Engineering Copilot for the contest {}.\n
                #         REQUIREMENTS: {}.\n
                #         ROOT SEARCH PATH: {}.\n
                #         f"DATA DESCRIPTION: {}.\n\n"
                #          "TASK: Write a Python script to Find, Inspect, and Prepare data.\n"


                #         "LOGIC FLOW IN DATA PROCESSING:\n"
                #         "1. SEARCH: Find the subfolder inside ROOT SEARCH PATH that matches the Contest Name.\n"
                #         "2. INSPECT: List all files in that folder.\n"
                #         "3. DECIDE STRATEGY:\n"
        
                #         "   - CASE: check in folder, if only one file csv => load this as data files.\n"
                #         "4. PREPARE: Handle missing values and Encode categoricals (OneHot/Label).\n"
                #         "5. DETECT: Check label and data, label can based on data_description, rename this col into 'target' "
                #         "6. OUTPUT: Save 'train.csv' and 'test.csv' to the current directory.\n"
                #         "7. PRINT: 'DATA_PROCESSED: 'train.csv' and 'test.csv' at the end.\n"
                #         "OUTPUT: ONLY valid Python code."
                #         """


                # system =  """Carefully read the following instructions to write Python code 
                #     {}
                #  """ 


                # system = system.format(code_instructions)
                # system = system.format(
                #     self.specs['name'],
                #     self.specs['requirements'],
                #     self.data_path,
                #     self.specs['data_description']
                #                     )
                print("____"*69)
                print("PROMPT GEN CODE \n")
                print(exec_prompt)

                res = call_llm(agent_profile, exec_prompt)
                raw_completion = res['response']
                completion = raw_completion.split("```python")[1].split("```")[0]
                self.money[f'Operation_Coding_{iteration}'] = {
                                                        "prompt_tokens": res.get("prompt_eval_count", 0),
                                                        "completion_tokens": res.get("eval_count", 0),
                                                        "total_tokens": res.get("prompt_eval_count", 0) + res.get("eval_count", 0)
                                                        }

                if not completion.strip(" \n"):
                    continue
                
                filename = f"{self.root_path}{self.code_path}.py"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "wt") as file:
                    file.write(completion)
                code = completion
                rcode, log = self.self_validation(filename)
                if rcode == 0:
                    action_result = log
                    break
                else:
                    log = log
                    error_logs.append(log)
                    action_result = log
                    print(self.agent_type, f"I got this error (itr #{iteration}): {log}")
                    iteration += 1                    
                    # break
            except Exception as e:
                iteration += 1
                print(self.agent_type, f"===== Retry: {iteration} =====")
                print(self.agent_type, f"Executioin error occurs: {e}")
            continue
        if not completion:
            completion = ""

        print(
            self.agent_type,
            f"I executed the given plan and got the follow results:\n\n{action_result}",
        )
        return {"rcode": rcode, "action_result": action_result, "code": completion, "error_logs": error_logs}


        #     except Exception as e:
        #         print(self.agent_type, f"===== Retry: {iteration} =====")
        #         print(self.agent_type, f"Executioin error occurs: {e}")
        #         iteration += 1
        #         continue
        #     if not completion:
        #         completion = ""
        #     return {"action_result": action_result, "code": completion, "error_logs": error_logs}
        # return {"action_result": action_result, "code": completion, "error_logs": error_logs}