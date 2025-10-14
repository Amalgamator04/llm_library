from typing import Literal

class LLMConnections:
    def __init__(self):
        # OpenRouter models list
        self.openrouter_models = [
            # Add your OpenRouter models here
             "openai/gpt-oss-20b:free",
            "z-ai/glm-4.5-air:free",
            "qwen/qwen3-coder:free",
            "moonshotai/kimi-k2:free",
            "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            "google/gemma-3n-e2b-it:free",
            "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            "tencent/hunyuan-a13b-instruct:free",
            "tngtech/deepseek-r1t2-chimera:free",
            "mistralai/mistral-small-3.2-24b-instruct:free",
            "moonshotai/kimi-dev-72b:free",
            "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "deepseek/deepseek-r1-0528:free",
            "sarvamai/sarvam-m:free",
            "mistralai/devstral-small-2505:free",
            "google/gemma-3n-e4b-it:free",
            "qwen/qwen3-4b:free",
            "qwen/qwen3-30b-a3b:free",
            "qwen/qwen3-8b:free",
            "qwen/qwen3-14b:free",
            "qwen/qwen3-235b-a22b:free",
            "tngtech/deepseek-r1t-chimera:free",
            "microsoft/mai-ds-r1:free",
            "shisa-ai/shisa-v2-llama3.3-70b:free",
            "arliai/qwq-32b-arliai-rpr-v1:free",
            "agentica-org/deepcoder-14b-preview:free",
            "moonshotai/kimi-vl-a3b-thinking:free",
            "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
            "google/gemini-2.5-pro-exp-03-25",
            "qwen/qwen2.5-vl-32b-instruct:free",
            "qwen/qwen2.5-vl-32b-instruct:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "featherless/qwerky-72b:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "google/gemma-3-4b-it:free",
            "rekaai/reka-flash-3:free",
            "google/gemma-3-27b-it:free",
            "qwen/qwq-32b:free",
            "nousresearch/deephermes-3-llama-3-8b-preview:free",
            "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
            "cognitivecomputations/dolphin3.0-mistral-24b:free",
            "cognitivecomputations/dolphin3.0-mistral-24b:free",
            "mistralai/mistral-small-24b-instruct-2501:free",
            "deepseek/deepseek-r1-distill-qwen-14b:free",
            "deepseek/deepseek-r1-distill-llama-70b:free",
            "deepseek/deepseek-r1:free",
            "google/gemini-2.0-flash-exp:free",
            "qwen/qwen-2.5-coder-32b-instruct:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "openrouter/sonoma-dusk-alpha",
            "openrouter/sonoma-sky-alpha"
        ]
        
        # SAM models organized by category
        self.sam_models = {
            "reasoning": [
                # Add your reasoning SAM models here
                "DeepSeek-R1-0528",
                "DeepSeek-R1-Distill-Llama-70B",
                "DeepSeek-V3.1",
                "Qwen3-32B"
            ],
            "only_text": [
                # Add your text-only SAM models here
                "DeepSeek-V3-0324",
                "Llama-3.3-Swallow-70B-Instruct-v0.4",
                "Meta-Llama-3.1-8B-Instruct",
                "Meta-Llama-3.3-70B-Instruct"
            ],
            "images": [
                # Add your image-capable SAM models here
                "Llama-4-Maverick-17B-128E-Instruct"
            ],
            "text_and_audio": [
                # Add your text and audio SAM models here
                "Whisper-Large-v3"
            ]
        }

    def list_models(self):
        """
        Return the list of all available models organized by category.
        
        Returns:
            dict: A dictionary containing openrouter_models and sam_models with their subcategories
        """
        return {
            "openrouter_models": self.openrouter_models,
            "sam_models": self.sam_models
        }

    def list_functions(self):
        """
        Return the list of all available function names in this class.
        
        Returns:
            list: A list of strings containing all available function names
        """
        return [
            "list_models",
            "call_open_free_llm", 
            "call_perplexity_llm",
            "call_huggingface_llm",
            "call_sam_reasoning_llm",
            "call_sam_text_llm",
            "call_sam_image_llm",
            "call_sam_audio_llm"
        ]

    def call_open_free_llm(self, llm_name: str, api_key: str, prompt: str):
        """
        Call OpenRouter API with the given model, token, and prompt.
        
        Parameters:
            llm_name (str): The name of the model to use (must be in the allowed models list)
            api_key (str): The OpenRouter API key for authentication
            prompt (str): The user prompt/message to send to the model
            
        Returns:
            str: The response content from the model
            
        Raises:
            ValueError: If the specified model is not in the allowed models list
        """
        if llm_name not in self.openrouter_models:
            raise ValueError(f"Model '{llm_name}' is not in the allowed OpenRouter model list.")

        from openai import OpenAI

        # Initialize OpenAI client with user-provided token
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Make the completion request
        completion = client.chat.completions.create(
            extra_body={},
            model=llm_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Return the response text
        return completion.choices[0].message.content

    def call_perplexity_llm(self, model_name: str, api_key: str, prompt: str, system_prompt: str = "You are a chatbot who answers the asked stuff in prompt.") -> str:
        """
        Call Perplexity Sonar LLM with web search capabilities.
        
        Parameters:
            model_name (str): The name of the Perplexity model to use (e.g., "sonar-pro")
            api_key (str): The Perplexity API key for authentication
            prompt (str): The user prompt/message to send to the model
            system_prompt (str, optional): The system message to set the model's behavior. 
                                         Defaults to "You are a chatbot who answers the asked stuff in prompt."
            
        Returns:
            str: The response content from the Perplexity model
            
        Note:
            This function enables web search mode by default for real-time information retrieval.
        """
        from openai import OpenAI

        # Initialize Perplexity API client
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            extra_body={"search_mode": "web"}  # optional extra params
        ) 
        # Ensure we always return a str (client may return None for content)
        content = getattr(resp.choices[0].message, "content", None)
        return content or ""

    def call_huggingface_llm(self, prompt: str, api_key: str, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"):
        """
        Call Hugging Face Inference API with the specified model.
        
        Parameters:
            prompt (str): The user prompt/message to send to the model
            api_key (str): The Hugging Face API token for authentication
            model_name (str, optional): The Hugging Face model to use. 
                                      Defaults to "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"
            
        Returns:
            str: The response content from the Hugging Face model
            
        Example:
            llm = LLMConnections()
            response = llm.call_huggingface_llm(
                prompt="What is the capital of France?",
                api_key="your-huggingface-token"
            )
            print(response)
        """
        from openai import OpenAI

        # Initialize Hugging Face client
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key
        )

        # Make the completion request
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return completion.choices[0].message.content

    def call_sam_reasoning_llm(self, api_key: str, model_name: str, prompt: str, system_prompt: str = "You are a helpful assistant", temperature: float = 0.1):
        """
        Call SambaNova SAM API for reasoning models.
        
        Parameters:
            api_key (str): The SambaNova API key for authentication
            model_name (str): The name of the SAM reasoning model to use
            prompt (str): The user's text prompt
            system_prompt (str, optional): The system message. Defaults to "You are a helpful assistant"
            temperature (float, optional): The sampling temperature. Defaults to 0.1
            
        Returns:
            str: The content of the generated message. Returns a message indicating an error if the API call fails
        """
        if not api_key:
            return "Error: API key is missing."
        if not model_name:
            return "Error: Model name is missing."
        if not prompt:
            return "Error: Prompt is missing."
        if model_name not in self.sam_models["reasoning"]:
            return f"Error: Model '{model_name}' is not in the allowed reasoning models list."

        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.sambanova.ai/v1",
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=0.1
            )
            
            return response.choices[0].message.content

        except Exception as e:
            return f"An error occurred: {e}"

    def call_sam_text_llm(self, api_key: str, model_name: str, prompt: str, system_prompt: str = "You are a helpful assistant", temperature: float = 0.1):
        """
        Call SambaNova SAM API for text-only models.
        
        Parameters:
            api_key (str): The SambaNova API key for authentication
            model_name (str): The name of the SAM text model to use
            prompt (str): The user's text prompt
            system_prompt (str, optional): The system message. Defaults to "You are a helpful assistant"
            temperature (float, optional): The sampling temperature. Defaults to 0.1
            
        Returns:
            str: The content of the generated message. Returns a message indicating an error if the API call fails
        """
        if not api_key:
            return "Error: API key is missing."
        if not model_name:
            return "Error: Model name is missing."
        if not prompt:
            return "Error: Prompt is missing."
        if model_name not in self.sam_models["only_text"]:
            return f"Error: Model '{model_name}' is not in the allowed text models list."

        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.sambanova.ai/v1",
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=0.1
            )
            
            return response.choices[0].message.content

        except Exception as e:
            return f"An error occurred: {e}"

    def call_sam_image_llm(self, api_key: str, model_name: str, prompt: str, image_url: str, temperature: float = 0.1):
        """
        Call SambaNova SAM API for image-capable models.
        
        Parameters:
            api_key (str): The SambaNova API key for authentication
            model_name (str): The name of the SAM image model to use
            prompt (str): The user's text prompt about the image
            image_url (str): A URL for an image, typically in base64 format
            temperature (float, optional): The sampling temperature. Defaults to 0.1
            
        Returns:
            str: The content of the generated message. Returns a message indicating an error if the API call fails
        """
        if not api_key:
            return "Error: API key is missing."
        if not model_name:
            return "Error: Model name is missing."
        if not prompt:
            return "Error: Prompt is missing."
        if not image_url:
            return "Error: Image URL is missing."
        if model_name not in self.sam_models["images"]:
            return f"Error: Model '{model_name}' is not in the allowed image models list."

        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.sambanova.ai/v1",
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }],
                temperature=temperature,
                top_p=0.1
            )
            
            return response.choices[0].message.content

        except Exception as e:
            return f"An error occurred: {e}"

    def call_sam_audio_llm(self, api_key: str, model_name: str, audio_file_path: str, response_format: Literal["text", "srt", "vtt"] = "text") -> str:
        """
        Call SambaNova SAM API for audio transcription models.

        Returns the transcribed text as a string. Handles several response shapes returned by clients.
        """
        if not api_key:
            return "Error: API key is missing."
        if not model_name:
            return "Error: Model name is missing."
        if not audio_file_path:
            return "Error: Audio file path is missing."
        if model_name not in self.sam_models["text_and_audio"]:
            return f"Error: Model '{model_name}' is not in the allowed audio models list."

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.sambanova.ai/v1",
            )

            with open(audio_file_path, "rb") as audio:
                response = client.audio.transcriptions.create(
                    model=model_name,
                    file=audio,
                    response_format=response_format,
                )

            # Normalize different possible response shapes to a string
            if isinstance(response, str):
                return response
            if isinstance(response, dict):
                return response.get("text", "") or ""
            # Some client objects expose `.text` attribute
            if hasattr(response, "text") and getattr(response, "text") is not None:
                return getattr(response, "text")
            return str(response)

        except Exception as e:
            return f"An error occurred: {e}"


