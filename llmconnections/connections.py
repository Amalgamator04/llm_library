class LLMConnections:
    def __init__(self):
        # OpenRouter models list
        self.openrouter_models = [
            # Add your OpenRouter models here
            "xiaomi/mimo-v2-flash:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "mistralai/devstral-2512:free",
            "nex-agi/deepseek-v3.1-nex-n1:free",
            "amazon/nova-2-lite-v1:free",
            "tngtech/tng-r1t-chimera:free",
            "allenai/olmo-3-32b-think:free",
            "kwaipilot/kat-coder-pro:free",
            "arcee-ai/trinity-mini:free",
            "nvidia/nemotron-nano-12b-v2-vl:free",
            "alibaba/tongyi-deepresearch-30b-a3b:free",
            "meituan/longcat-flash-chat:free",
            "nvidia/nemotron-nano-9b-v2:free",
            "openai/gpt-oss-20b:free",
            "z-ai/glm-4.5-air:free",
            "qwen/qwen3-coder:free",
            "moonshotai/kimi-k2:free",
            "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            "google/gemma-3n-e2b-it:free",
            "tngtech/deepseek-r1t2-chimera:free",
            "google/gemma-3n-e4b-it:free",
            "qwen/qwen3-4b:free",
            "qwen/qwen3-235b-a22b:free",
            "tngtech/deepseek-r1t-chimera:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "google/gemma-3-4b-it:free",
            "google/gemma-3-12b-it:free",
            "google/gemma-3-27b-it:free",
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "mistralai/mistral-7b-instruct:free"
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
            "call_sam_audio_llm",
            "call_cohere_embed",
            "call_glm_llm",
            "safe_to_dict",
            "get_embedding"
        ]
   



    def get_embedding(text: str) -> list[float]:
        import requests
        OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
        MODEL_NAME = "nomic-embed-text"
        """
        Generate embedding using local Ollama model.
    
        Args:
            text (str): Input text
    
        Returns:
            list[float]: Vector embedding
        ITS HAS 768 vector length
        """
        payload = {
            "model": MODEL_NAME,
            "prompt": text
        }
    
        response = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=30)
        response.raise_for_status()
    
        return response.json()["embedding"]
    
    def safe_to_dict(self, text: str):
        import re, json
        try:
            # Extract the first JSON object from the text using regex
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match is None:
                return {"error": "No JSON object found in text"}
            json_str = match.group()
            return json.loads(json_str)
        except Exception as e:
            return {"error": f"Invalid JSON or parsing failed: {e}"}
        
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
        return resp.choices[0].message.content

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

    def call_sam_audio_llm(self, api_key: str, model_name: str, audio_file_path: str, response_format: str = "text"):
        """
        Call SambaNova SAM API for audio transcription models.
        
        Parameters:
            api_key (str): The SambaNova API key for authentication
            model_name (str): The name of the SAM audio model to use
            audio_file_path (str): Path to the audio file to transcribe
            response_format (str, optional): The response format. Defaults to "text"
            
        Returns:
            str: The transcribed text. Returns a message indicating an error if the API call fails
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
            return response

        except Exception as e:
            return f"An error occurred: {e}"

    def call_cohere_embed(self, api_key: str, texts: list, model: str = "embed-english-v3.0", input_type: str = "classification", truncate: str = "NONE"):
        """
        Call Cohere Embedding API to generate embeddings for texts.
        
        Parameters:
            api_key (str): The Cohere API key for authentication
            texts (list): A list of text strings to generate embeddings for
            model (str, optional): The Cohere embedding model to use. 
                                 Defaults to "embed-english-v3.0"
            input_type (str, optional): The type of input for embeddings. 
                                      Defaults to "classification"
            truncate (str, optional): How to handle text truncation. 
                                    Defaults to "NONE"
            
        Returns:
            list: The embeddings list. Returns an error message if the API call fails
            
        Example:
            llm = LLMConnections()
            texts = ["Hello world", "How are you?"]
            embeddings = llm.call_cohere_embed(
                api_key="your-cohere-api-key",
                texts=texts,
                model="embed-english-v3.0",
                input_type="classification"
            )
            print(embeddings)
        """
        if not api_key:
            return {"error": "API key is missing."}
        if not texts:
            return {"error": "Texts list is empty."}
        if not isinstance(texts, list):
            return {"error": "Texts must be a list."}
        
        try:
            import cohere
            
            co = cohere.Client(api_key)
            
            response = co.embed(
                model=model,
                texts=texts,
                input_type=input_type,
                truncate=truncate
            )
            
            return response.embeddings
        
        except Exception as e:
            return {"error": f"An error occurred: {e}"}

    def call_glm_llm(self, api_key: str, prompt: str, model: str = "GLM-4.5-Flash", system_prompt: str = "You are a useful AI assistant.", temperature: float = 0.6, max_tokens: int = 1024):
        """
        Call GLM API for chat completions.
        
        Parameters:
            api_key (str): The GLM API key for authentication
            prompt (str): The user's message/prompt
            model (str, optional): The GLM model to use. Defaults to "GLM-4.5-Flash"
            system_prompt (str, optional): The system message. Defaults to "You are a useful AI assistant."
            temperature (float, optional): The sampling temperature. Defaults to 0.6
            max_tokens (int, optional): Maximum tokens in response. Defaults to 1024
            
        Returns:
            str: The response content from the GLM model. Returns an error message if the API call fails
            
        Example:
            llm = LLMConnections()
            response = llm.call_glm_llm(
                api_key="your-glm-api-key",
                prompt="Please tell us about artificial intelligence."
            )
            print(response)
        """
        if not api_key:
            return "Error: API key is missing."
        if not prompt:
            return "Error: Prompt is missing."
        
        try:
            import requests
            
            url = "https://api.z.ai/api/paas/v4/chat/completions"
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API request failed with status code {response.status_code}: {response.text}"}
        
        except Exception as e:
            return {"error": f"An error occurred: {e}"}


