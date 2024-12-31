import os
import yaml
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import OpenAI
from langchain_openai import AzureOpenAI, OpenAIEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config_path):
        """
        Initialize ModelManager by loading configuration and setting up models.
        :param config_path: Path to the YAML configuration file.
        """
        load_dotenv()
        
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully from %s", config_path)
        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            raise

        self.llm = None
        self.embedding_model = None
        
        if self.config['llm_provider'] == 'groq':
            self.load_llm_groq()
            
        if self.config['llm_provider'] == 'ollama':
            self.load_llm_ollama()
            
        if self.config['llm_provider'] == 'openai':
            self.load_llm_openai()
        
        
    # def get_llm_response(self):
    #     if self.config['llm_provider'] == 'groq':
    #         return self.llm_groq.invoke(prompt).content
    #     if self.config['llm_provider'] == 'ollama':
    #         return self.llm_ollama(prompt)
    #     if self.config['llm_provider'] == 'openai':
    #         return self.llm_openai(prompt)

    def load_llm_groq(self):
        """
        Lazily loads the LLM Groq model based on the configuration if it hasn't been loaded yet.
        :return: The loaded LLM Groq model.
        """
        if not self.llm:
            try:
                logger.info("Loading Groq LLM model...")
                self.llm = ChatGroq(
                    model_name=self.config['llm_model'],
                    api_key=os.getenv("GROQ_API_KEY")
                )
                logger.info("Groq LLM model loaded successfully.")
                
            except KeyError as e:
                logger.error("Missing required config key for LLM: %s", e)
                raise
            except Exception as e:
                logger.error("Error loading Groq LLM model: %s", e)
                raise
    
    def load_llm_openai(self):
        """
        Loads the OpenAI model.
        :return: The loaded OpenAI model.
        """
        if not self.llm:
            try:
                logger.info("Loading OpenAI model...")
                # openai.api_key = os.getenv("OPENAI_API_KEY")
                # os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
                # os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
                # os.environ["AZURE_OPENAI_API_KEY"] = "..."
                print("AZURE_OPENAI_DEPLOYMENT:", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
                load_dotenv()
                # print("AZURE_OPENAI_DEPLOYMENT:", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
                self.llm = AzureOpenAI(
                    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # The deployment name of the model
                    )
                logger.info("OpenAI model loaded successfully.")
                
            except KeyError as e:
                logger.error("Missing required config key for OpenAI LLM: %s", e)
                raise
            except Exception as e:
                logger.error("Error loading OpenAI LLM model: %s", e)
                raise
            
    def load_llm_ollama(self):
        """
        Loads the Ollama LLM model.
        :return: The loaded Ollama LLM model.
        """
        if not self.llm:
            try:
                logger.info("Loading Ollama LLM model...")
                
                self.llm = ChatOllama(
                                model=self.config['llm_model'],
                                temperature=0,
                                )
                logger.info("Ollama LLM model loaded successfully.")
                          
            except KeyError as e:
                logger.error("Missing required config key for Ollama LLM: %s", e)
                raise
            except Exception as e:
                logger.error("Error loading Ollama LLM model: %s", e)
                raise

    def load_embedding_model(self):
        """
        Lazily Loads the embedding model based on the cloud provider configuration. if it hasn't been loaded yet.
        :return: The loaded embedding model.
        """
        if not self.embedding_model:
            try:
                logger.info("Loading embedding model...")
                if self.config['cloud_provider'] == 'ollama':
                    self.embedding_model = OllamaEmbeddings(
                        model=self.config['embedding_model']
                    )
                elif self.config['cloud_provider'] == 'groq':
                    # Since Groq doesn't have embeddings yet, we'll use OpenAI's
                    self.embedding_model = OpenAIEmbeddings(
                        model=self.config['embedding_model']
                    )

                logger.info("Embedding model loaded successfully.")
            except KeyError as e:
                logger.error("Missing required config key for embedding model: %s", e)
                raise
            except Exception as e:
                logger.error("Error loading embedding model: %s", e)
                raise
        return self.embedding_model

    def count_tokens(self, text):
        """
        Counts the number of tokens in the given text using the Groq LLM.
        :param text: The text to count tokens for.
        :return: Number of tokens in the text.
        """
        if not self.llm:
            logger.warning("Groq LLM model is not loaded. Loading the model first...")
            self.load_llm_groq()
        
        try:
            num_tokens = self.llm.get_num_tokens(text)
            logger.info("Successfully counted %d tokens for the given text.", num_tokens)
            return num_tokens
        except Exception as e:
            logger.error("Error counting tokens: %s", e)
            raise


# Main function for testing the ModelManager class
if __name__ == '__main__':
    try:
        model_manager = ModelManager('config/config.yaml')
        model_manager.load_embedding_model()

        # print("Token count:", model_manager.count_tokens("Hello world."))
        print("Groq LLM Response:", model_manager.llm.invoke("Hello world."))
        print("Embedding Result:", model_manager.embedding_model.embed_documents(["Hello world."]))
        # print("LLM content:", model_manager.get_llm_response("Hello world."))
    except Exception as e:
        logger.error("An error occurred: %s", e)