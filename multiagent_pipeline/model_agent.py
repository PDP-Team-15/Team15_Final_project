from openai import OpenAI
import logging
import os
import re
from utils import detect_suspicious_code, save_suspicious_example

class ModelAgent:
    """
    Single local model (Qwen 14B AWQ) used for both translation and fixing.
    translator_client and fixer_client point to the same vLLM server so the
    32768-token context window is available for all stages.
    """
    def __init__(self,
                 translator_base_url='http://localhost:8004/v1',
                 translator_model='Qwen/Qwen2.5-Coder-14B-Instruct-AWQ',
                 fixer_base_url='http://localhost:8004/v1',
                 fixer_model='Qwen/Qwen2.5-Coder-14B-Instruct-AWQ',
                 api_key='token123',
                 max_tokens=15000,
                 temperature=0.2,
                 top_p=0.9,
                 # legacy params accepted but unused
                 model=None, api_base=None, openai_model=None):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.translator_model = translator_model
        self.fixer_model = fixer_model

        self.translator_client = OpenAI(
            base_url=translator_base_url,
            api_key=api_key,
        )
        self.fixer_client = OpenAI(
            base_url=fixer_base_url,
            api_key=api_key,
        )

        logging.info(
            f"ModelAgent initialized: "
            f"translator={translator_model}@{translator_base_url}, "
            f"fixer={fixer_model}@{fixer_base_url}"
        )

    def generate_translation(self, messages, num_return_sequences=1, context=None):
        """
        Generate code translation based on provided messages.
        Returns the translated code as a string, or None if the example should be skipped.

        Args:
            messages: The messages to send to the model
            num_return_sequences: Number of responses to generate
            context: Additional context about the code (e.g., kernel_name, from_api, to_api)

        Returns:
            str: The translated code, or None if the example should be skipped
        """
        try:
            logging.info(f"Calling translator model with temperature={self.temperature}, top_p={self.top_p}, max_tokens={self.max_tokens}")
            response = self.translator_client.chat.completions.create(
                model=self.translator_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                n=num_return_sequences
            )

            # Return the first response's content
            if response.choices:
                generated_code = response.choices[0].message.content

                # Save the full model response if context is provided
                if context and 'kernel_name' in context:
                    kernel_name = context['kernel_name']
                    from_api = context.get('from_api', 'unknown')
                    to_api = context.get('to_api', 'unknown')

                    # Get the current results folder from context or use default
                    current_results_folder = context.get('current_results_folder', 'kernel_logs')

                    # Create a directory for this kernel if it doesn't exist
                    kernel_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                 'output', current_results_folder, f"{kernel_name}_{from_api}_to_{to_api}")
                    os.makedirs(kernel_log_dir, exist_ok=True)

                    # Save the model response
                    with open(os.path.join(kernel_log_dir, f"translation_response.txt"), "w") as f:
                        f.write(f"Model: {self.translator_model}\n")
                        f.write(f"Temperature: {self.temperature}\n")
                        f.write(f"Top_p: {self.top_p}\n")
                        f.write(f"Max tokens: {self.max_tokens}\n\n")
                        f.write("=== PROMPT ===\n")
                        for msg in messages:
                            f.write(f"{msg['role']}: {msg['content']}\n\n")
                        f.write("=== RESPONSE ===\n")
                        f.write(generated_code)

                    logging.info(f"Saved translation model response for kernel {kernel_name} to {kernel_log_dir}")

                # Extract code from the model's response
                from utils import extract_code_from_model_output
                extracted_code = extract_code_from_model_output(generated_code)

                # Check for suspicious or incomplete code
                is_suspicious, reason = detect_suspicious_code(extracted_code, context)
                if is_suspicious:
                    # Flag and save the suspicious example
                    logging.warning(f"Suspicious code detected: {reason}")
                    save_suspicious_example(extracted_code, reason, context)

                    # Return None to indicate that this example should be skipped
                    return None

                return generated_code
            else:
                logging.warning("Empty response from model")
                if context:
                    save_suspicious_example("", "Empty response from model", context)
                return None

        except Exception as e:
            logging.error(f"Error in generate_translation: {str(e)}")
            if context:
                save_suspicious_example("", f"Error in generate_translation: {str(e)}", context)
            # Return None to indicate that this example should be skipped
            # instead of raising an exception that would halt the entire experiment
            return None

    ## Agent for compilation agent logic
    def fix_code(self, code, error_message, to_api, context=None):
        """
        Send code with compilation error back to the model to fix.
        Returns the fixed code as a string, or None if the example should be skipped.

        Args:
            code: The code with compilation errors
            error_message: The compilation error message
            to_api: The target API (cuda or omp)
            context: Additional context about the code (e.g., kernel_name, from_api, to_api)

        Returns:
            str: The fixed code, or None if the example should be skipped
        """
        if to_api == 'cuda':
            speciality = "CUDA"
        elif to_api == 'omp':
            speciality = "C++ (OpenMP)"

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert {speciality} programmer specializing in HPC code. "
                    f"Your ONLY task is to fix compilation errors. "
                    f"STRICT RULES: "
                    f"(1) Make the MINIMUM changes necessary to resolve the compilation error. "
                    f"(2) Do NOT change any algorithm, computational logic, or numerical behavior. "
                    f"(3) Do NOT rename variables, change loop bounds, or alter data structures. "
                    f"(4) Do NOT add, remove, or reorder any functional code beyond what is required to fix the error. "
                    f"(5) If a platform-specific intrinsic is unavailable, replace it with a functionally identical alternative that produces the exact same numerical result. "
                    f"Provide only the complete corrected code inside a proper code block. No explanations, no omissions, no placeholders."
                )
            },
            {
                "role": "user",
                "content": f"The following code fails to compile with the given error message.\n\nCode:\n```\n{code}\n```\n\nCompilation Error:\n```\n{error_message}\n```\n\nFix ONLY the compilation error. Do NOT change the algorithm or logic. Output only the complete corrected code inside:\n```{to_api}\n<full corrected code here>\n```"
            },
            {
                "role": "assistant",
                "content": "```cpp\n"
            }
        ]

        try:
            logging.info(f"Calling fixer model with temperature={self.temperature}, top_p={self.top_p}, max_tokens={self.max_tokens}")
            response = self.fixer_client.chat.completions.create(
                model=self.fixer_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )

            # Return the first response's content
            if response.choices:
                fixed_code = response.choices[0].message.content

                # Save the full model response if context is provided
                if context and 'kernel_name' in context:
                    kernel_name = context['kernel_name']
                    from_api = context.get('from_api', 'unknown')
                    to_api = context.get('to_api', 'unknown')
                    attempt = context.get('attempt', 0)

                    # Get the current results folder from context or use default
                    current_results_folder = context.get('current_results_folder', 'kernel_logs')

                    # Create a directory for this kernel if it doesn't exist
                    kernel_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                 'output', current_results_folder, f"{kernel_name}_{from_api}_to_{to_api}")
                    os.makedirs(kernel_log_dir, exist_ok=True)

                    # Save the model response with attempt number
                    with open(os.path.join(kernel_log_dir, f"compilation_fix_attempt_{attempt}.txt"), "w") as f:
                        f.write(f"Model: {self.fixer_model}\n")
                        f.write(f"Temperature: {self.temperature}\n")
                        f.write(f"Top_p: {self.top_p}\n")
                        f.write(f"Max tokens: {self.max_tokens}\n\n")
                        f.write("=== ERROR MESSAGE ===\n")
                        f.write(error_message)
                        f.write("\n\n=== PROMPT ===\n")
                        for msg in messages:
                            f.write(f"{msg['role']}: {msg['content']}\n\n")
                        f.write("=== RESPONSE ===\n")
                        f.write(fixed_code)

                    logging.info(f"Saved compilation fix model response for kernel {kernel_name} (attempt {attempt}) to {kernel_log_dir}")

                    # Also save to the attempt_compile_X directory if it exists
                    next_attempt = attempt + 1
                    compilation_dir = os.path.join(os.path.dirname(kernel_log_dir), 
                                                 "compilation", 
                                                 f"{kernel_name}_{from_api}_{to_api}")

                    if os.path.exists(compilation_dir):
                        # Look for the attempt_compile_{next_attempt} directory
                        next_attempt_dir = os.path.join(compilation_dir, f"attempt_compile_{next_attempt}")

                        if os.path.exists(next_attempt_dir):
                            # Save the model response file
                            with open(os.path.join(next_attempt_dir, "model_response.txt"), "w") as f:
                                f.write(fixed_code)
                            logging.info(f"Also saved model response to {next_attempt_dir}")
                        else:
                            # If directory doesn't exist, create a new one
                            next_attempt_dir = os.path.join(compilation_dir, f"attempt_compile_{next_attempt}")
                            os.makedirs(next_attempt_dir, exist_ok=True)

                            # Save the model response file
                            with open(os.path.join(next_attempt_dir, "model_response.txt"), "w") as f:
                                f.write(fixed_code)
                            logging.info(f"Created new directory and saved model response to {next_attempt_dir}")

                return fixed_code

            else:
                logging.warning("Empty response from model when fixing code")
                return code  # Return original code if no response

        except Exception as e:
            err_str = str(e)
            logging.error(f"Error in fix_code: {err_str}")
            if 'Error code: 400' in err_str and 'maximum context length' in err_str:
                m = re.search(r'maximum context length is (\d+).*?(\d+) in the messages', err_str, re.DOTALL)
                if m:
                    max_ctx = int(m.group(1))
                    msg_tokens = int(m.group(2))
                    reduced_max = max(500, max_ctx - msg_tokens - 50)
                    logging.warning(f"Context window exceeded, retrying fix_code with max_tokens={reduced_max}")
                    try:
                        response = self.fixer_client.chat.completions.create(
                            model=self.fixer_model,
                            messages=messages,
                            max_tokens=reduced_max,
                            temperature=self.temperature,
                            top_p=self.top_p
                        )
                        if response.choices:
                            return response.choices[0].message.content
                    except Exception as e2:
                        logging.error(f"Retry with reduced max_tokens also failed: {e2}")
            if 'Error code: 400' in err_str:
                return code + '\n\n' + err_str
            return code

    def run_code(self, code, error_message, input, to_api, context=None):
        """
        Send code with runtime error back to the model to fix.
        Returns the fixed code or the original code if there are issues, but never returns None.

        Args:
            code: The code with runtime errors
            error_message: The runtime error message
            input: The input that caused the error
            to_api: The target API (cuda or omp)
            context: Additional context about the code (e.g., kernel_name, from_api, to_api)

        Returns:
            str: The fixed code, or the original code if there are issues
        """
        if to_api == 'cuda':
            speciality = "CUDA"
        elif to_api == 'omp':
            speciality = "C++ (OpenMP)"

        messages = [
            {
                "role": "system",
                "content": f"You are an expert {speciality} programmer specializing in High-Performance Computing (HPC). Fix runtime errors such as segmentation faults, timeouts, or illegal memory accesses. Output only the full corrected code without explanations or placeholders. Do not omit anything."
            },
            {
                "role": "user",
                "content": f"The following code is running on {input} and compiles successfully but produces a runtime error: {error_message}.\n\nPlease fix it. Output only the corrected full code. No explanations, no comments, no omissions. DON'T CHANGE THE main FUNCTION!\n\n```{to_api}\n{code}\n```"
            },
            {
                "role": "assistant",
                "content": f"```{to_api}\n// Corrected code will be here\n```"
            }
        ]

        try:
            logging.info(f"Calling fixer model with temperature={self.temperature}, top_p={self.top_p}, max_tokens={self.max_tokens}")
            response = self.fixer_client.chat.completions.create(
                model=self.fixer_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )

            # Return the first response's content
            if response.choices:
                fixed_code = response.choices[0].message.content

                # Save the full model response if context is provided
                if context and 'kernel_name' in context:
                    kernel_name = context['kernel_name']
                    from_api = context.get('from_api', 'unknown')
                    to_api = context.get('to_api', 'unknown')
                    attempt = context.get('attempt', 0)

                    # Get the current results folder from context or use default
                    current_results_folder = context.get('current_results_folder', 'kernel_logs')

                    # Create a directory for this kernel if it doesn't exist
                    kernel_log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                 'output', current_results_folder, f"{kernel_name}_{from_api}_to_{to_api}")
                    os.makedirs(kernel_log_dir, exist_ok=True)

                    # Save the model response with attempt number
                    with open(os.path.join(kernel_log_dir, f"runtime_fix_attempt_{attempt}.txt"), "w") as f:
                        f.write(f"Model: {self.fixer_model}\n")
                        f.write(f"Temperature: {self.temperature}\n")
                        f.write(f"Top_p: {self.top_p}\n")
                        f.write(f"Max tokens: {self.max_tokens}\n\n")
                        f.write("=== ERROR MESSAGE ===\n")
                        f.write(error_message)
                        f.write("\n\n=== INPUT ===\n")
                        f.write(input)
                        f.write("\n\n=== PROMPT ===\n")
                        for msg in messages:
                            f.write(f"{msg['role']}: {msg['content']}\n\n")
                        f.write("=== RESPONSE ===\n")
                        f.write(fixed_code)

                    logging.info(f"Saved runtime fix model response for kernel {kernel_name} (attempt {attempt}) to {kernel_log_dir}")
                   

                return fixed_code
            else:
                logging.warning("Empty response from model when fixing runtime code")
                return code  # Return original code if no response

        except Exception as e:
            err_str = str(e)
            logging.error(f"Error in run_code: {err_str}")
            if 'Error code: 400' in err_str and 'maximum context length' in err_str:
                m = re.search(r'maximum context length is (\d+).*?(\d+) in the messages', err_str, re.DOTALL)
                if m:
                    max_ctx = int(m.group(1))
                    msg_tokens = int(m.group(2))
                    reduced_max = max(500, max_ctx - msg_tokens - 50)
                    logging.warning(f"Context window exceeded, retrying run_code with max_tokens={reduced_max}")
                    try:
                        response = self.fixer_client.chat.completions.create(
                            model=self.fixer_model,
                            messages=messages,
                            max_tokens=reduced_max,
                            temperature=self.temperature,
                            top_p=self.top_p
                        )
                        if response.choices:
                            return response.choices[0].message.content
                    except Exception as e2:
                        logging.error(f"Retry with reduced max_tokens also failed: {e2}")
            if 'Error code: 400' in err_str:
                return code + '\n\n' + err_str
            return code
