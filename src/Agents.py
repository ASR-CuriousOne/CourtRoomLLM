import os
from groq import Groq
import pandas as pd
from dotenv import load_dotenv

# Load environment variables (Groq API key)
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Base Agent Class
class Agent:
    def __init__(self, role,summary_case = '' ,personality = None ,model="meta-llama/llama-4-maverick-17b-128e-instruct"):
        self.role = role
        self.model = model
        self.case_summary = summary_case
        self.personality = personality

        

    def generate_response(self, prompt, temperature=0.7, max_tokens=500):
        """Generate a response using the Groq API."""

        if(self.personality != None):
            prompt += f" Personality and style of response should be of following: {self.personality}. \
                Only text as roleplay should be included."

        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=temperature,
                max_completion_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response for {self.role}: {str(e)}"

# Role-Specific Agent Classes
class Defendant(Agent):
    def testify(self):
        prompt = f"As the Defendant in the case '{self.case_summary}', provide your testimony defending yourself."
        return self.generate_response(prompt)

class Plaintiff(Agent):
    def testify(self):
        prompt = f"As the Plaintiff in the case '{self.case_summary}', provide your testimony supporting your accusation."
        return self.generate_response(prompt)

class Lawyer(Agent):
    def make_statement(self, phase):
        prompt = f"As the {self.role} in the case '{self.case_summary}', provide a {phase} statement."
        return self.generate_response(prompt)

    def interrogate_witness(self, witness_statement):
        prompt = f"As the {self.role}, interrogate the witness based on their statement: '{witness_statement}'."
        return self.generate_response(prompt)

class Judge(Agent):
    def deliberate(self, trial_log):
        prompt = (
            f"As the Judge, review the following trial log: {trial_log}. "
            f"Provide a clear verdict (e.g., 'Guilty', 'Not Guilty', 'In Favor of Plaintiff', or 'In Favor of Defendant') "
            f"followed by a concise reasoning (150–250 words) explaining the decision. "
            f"The reasoning must be based solely on the evidence, witness statements, and applicable legal principles presented in the trial log, "
            f"and it should address the key factors influencing the verdict in a logical and impartial manner. "
            f"Use a formal, authoritative, and neutral tone, avoiding bias or emotional language. "
            f"Do not include introductory text or extraneous commentary beyond the verdict and reasoning."
            f"For verdict give a proper sentence"
        )
        return self.generate_response(prompt)

class Witness(Agent):
    def testify(self, context=""):
        prompt = f"As a Witness in the case '{self.case_summary}', provide testimony based on this context: '{context}'."
        return self.generate_response(prompt)

# Central Controller
class CourtroomSimulation:
    def __init__(self, case_descriptions,log_file_path,
                 plantiff: Plaintiff,
                 prosecution_lawyer: Lawyer,
                 defendant: Defendant,
                 defence_lawyer: Lawyer,
                 judge: Judge):
        
        self.case_descriptions = case_descriptions
        self.trial_log = []
        self.agents = {
            "defendant": defendant,
            "plaintiff": plantiff,
            "defense_lawyer": defence_lawyer,
            "prosecution_lawyer": prosecution_lawyer,
            "judge": judge
        }
        self.witnesses = []
        self.log_file_path = log_file_path

    def log(self, entry):
        """Add an entry to the trial log and print it."""
        print(entry)
        print("-----")
        self.trial_log.append(entry)

    def run_opening_statements(self):
        self.log("\n--- Opening Statements ---")
        for lawyer in ["prosecution_lawyer", "defense_lawyer"]:
            statement = self.agents[lawyer].make_statement("opening")
            self.log(f"{lawyer.replace('_', ' ').title()}: {statement}")

    def run_interrogation(self):
        self.log("\n--- Witness Interrogation & Argumentation ---")
        # Plaintiff and Defendant testify
        plaintiff_testimony = self.agents["plaintiff"].testify()
        self.log(f"Plaintiff Testimony: {plaintiff_testimony}")
        defendant_testimony = self.agents["defendant"].testify()
        self.log(f"Defendant Testimony: {defendant_testimony}")

        # Lawyers interrogate (example with static witness for simplicity)
        witness = Witness("Witness",self.case_descriptions["defendant"])
        witness_statement = witness.testify( "I saw the incident occur.")
        self.log(f"Witness Testimony: {witness_statement}")
        self.witnesses.append(witness)

        for lawyer in ["prosecution_lawyer", "defense_lawyer"]:
            interrogation = self.agents[lawyer].interrogate_witness(witness_statement)
            self.log(f"{lawyer.replace('_', ' ').title()} Interrogation: {interrogation}")

    def run_closing_statements(self):
        self.log("\n--- Closing Statements ---")
        for lawyer in ["prosecution_lawyer", "defense_lawyer"]:
            statement = self.agents[lawyer].make_statement("closing")
            self.log(f"{lawyer.replace('_', ' ').title()}: {statement}")

    def run_ruling(self):
        self.log("\n--- Judge’s Ruling ---")
        verdict = self.agents["judge"].deliberate("\n".join(self.trial_log))
        self.log(f"Judge: {verdict}")

    def simulate_trial(self):
        """Run the full trial simulation."""
        self.run_opening_statements()
        self.run_interrogation()
        self.run_closing_statements()
        self.run_ruling()
        return "\n".join(self.trial_log)
