class ModelAgent:
    def __init__(self, specs):
        self.target_metric = specs['target_metric']
        self.train_file = "processed_train.csv"
        self.test_file = "processed_test.csv"
        self.update_system_prompt()

        # self.system = (
        #     f"You are an Autonomous ML Researcher. Metric: {self.target_metric}.\n"
        #     "TASK: Train a model on 'processed_train.csv'.\n"
        #     "REQUIREMENTS:\n"
        #     "1. Load 'processed_train.csv'.\n"
        #     "2. Train a model (RandomForest/XGBoost).\n"
        #     "3. Internal Validation: Split processed_train 80/20 to calculate metric.\n"
        #     "4. PRINT FORMAT: 'FINAL_METRIC: <number>'.\n"
        #     "5. Prediction: Load 'processed_test.csv', predict, and save 'submission.csv'.\n"
        #     "OUTPUT: ONLY valid Python code."
        # )

    def update_system_prompt(self):
        self.system = (
            f"You are an Autonomous ML Researcher. Metric: {self.target_metric}.\n"
            f"TASK: Train a model on '{self.train_file}'.\n"
            "REQUIREMENTS:\n"
            f"1. Load '{self.train_file}'.\n"
            "2. Train a model (RandomForest/XGBoost).\n"
            f"3. Internal Validation: Split {self.train_file} 80/20 to calculate metric.\n"
            "4. PRINT FORMAT: 'FINAL_METRIC: <number>'.\n"
            f"5. Prediction: Load '{self.test_file}', predict, and save 'submission.csv'.\n"
            "OUTPUT: ONLY valid Python code."
        )

    # def generate(self, feedback=""):
    #     prompt = f"Generate training script optimizing for {self.target_metric}."
    #     if feedback:
    #         prompt += f"\nFIX ERROR: {feedback}"
    #     return call_llm(self.system, prompt)

    def set_data_files(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.update_system_prompt()

    def generate(self, feedback=""):
        prompt = f"Generate training script optimizing for {self.target_metric}."
        if feedback:
            prompt += f"\n\n⚠️ PREVIOUS RUN FAILED. FIX ERROR:\n{feedback}\n"
            prompt += "Review the code logic carefully. Output ONLY valid Python code."
        return call_llm(self.system, prompt)

