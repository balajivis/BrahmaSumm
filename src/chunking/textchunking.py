import re
import yaml
import logging

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkManager:
    """
    The ChunkManager class is responsible for dividing large text documents into chunks
    based on word count, while preserving sentence and paragraph boundaries.
    It ensures that chunks are created flexibly based on a target word count, 
    allowing for slight variations in chunk size.
    """

    def __init__(self, config_path):
        """
        Initializes ChunkManager by loading the configuration.

        :param config_path: Path to the configuration file containing chunking parameters.
        """
        logger.info("Initializing ChunkManager with configuration from %s", config_path)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.flexibility = self.config.get('chunk_flexibility', 0.25)  # Default to 25% flexibility
        self.target_words = self.config.get('target_words', 100)       # Default to 100 words per chunk
        self.chunks = []
        logger.info("ChunkManager initialized with target_words=%d and flexibility=%.2f", self.target_words, self.flexibility)

    def preprocess_text(self, text):
        """
        Cleans the input text by replacing multiple newlines and removing non-ASCII characters.
        
        :param text: The raw text to be preprocessed.
        :return: Cleaned text.
        """
        logger.info("Preprocessing text: removing extra newlines and non-ASCII characters")
        text = re.sub(r'\n+', '\n\n', text)
        text = re.sub(r'\t+', '\t', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        logger.debug("Preprocessed text: %s", text[:100])  # Log a preview of the preprocessed text
        return text

    def flexible_chunk(self, text, target_words=None, flexibility=None):
        """
        Divides the text into chunks based on a target word count and flexible chunk size.
        Ensures that each chunk contains a minimum of 75% and a maximum of 125% of the target word count.

        :param text: The preprocessed text to be chunked.
        :param target_words: Optional; target word count per chunk. If not provided, the default is used.
        :param flexibility: Optional; the percentage flexibility for chunk size. If not provided, the default is used.
        """
        
        if target_words is None:
            target_words = self.target_words
        if flexibility is None:
            flexibility = self.flexibility
            
        # coutn words in text
        text_word_count = len(text.split())        
        if text_word_count < 100:
            logger.warning("Text is too short to be chunked")
            target_words = 25
            
        elif text_word_count < 800:
            target_words = 50
            
        print("Target words",text_word_count, target_words, flexibility)

        min_words = int(target_words * (1 - flexibility))  # Minimum words per chunk (75%)
        max_words = int(target_words * (1 + flexibility))  # Maximum words per chunk (125%)

        logger.info("Chunking text with target_words=%d, flexibility=%.2f, min_words=%d, max_words=%d", 
                    target_words, flexibility, min_words, max_words)

        paragraphs = re.split(r'\n\n', text)  # Split by paragraphs
        chunks = []
        current_chunk = []
        current_word_count = 0

        def finalize_chunk(force=False):
            """Finalizes the current chunk if it meets the minimum word count or if forced."""
            if current_chunk and (force or current_word_count >= min_words):
                logger.debug("Finalizing chunk with %d words", current_word_count)
                chunks.append(' '.join(current_chunk))

        def process_paragraph(paragraph):
            """Processes a paragraph, splitting it into sentences and adding to chunks."""
            nonlocal current_chunk, current_word_count
            sentences = re.split(r'(?<=[.!?]) +', paragraph)  # Split by sentence
            for sentence in sentences:
                sentence_word_count = len(sentence.split())

                # If adding the sentence keeps the chunk under the max limit, add it
                if current_word_count + sentence_word_count <= max_words:
                    current_chunk.append(sentence)
                    current_word_count += sentence_word_count

                # If the chunk is at or above the minimum, finalize it
                if current_word_count >= min_words:
                    finalize_chunk()
                    current_chunk = []
                    current_word_count = 0

                # If adding the sentence exceeds the max, finalize and start a new chunk
                elif current_word_count + sentence_word_count > max_words:
                    finalize_chunk()
                    current_chunk = [sentence]
                    current_word_count = sentence_word_count

        # Process each paragraph and chunk the text accordingly
        for paragraph in paragraphs:
            para_word_count = len(paragraph.split())
            logger.debug("Processing paragraph with %d words", para_word_count)

            # If the paragraph itself is smaller than the max size, add it as a chunk
            if para_word_count <= max_words:
                current_chunk.append(paragraph)
                current_word_count += para_word_count
                if current_word_count >= min_words:
                    finalize_chunk()
                    current_chunk = []
                    current_word_count = 0
            else:
                process_paragraph(paragraph)

        # Finalize any remaining text as the last chunk
        finalize_chunk(force=True)
        self.chunks = chunks
        logger.info("Chunking completed with %d chunks", len(chunks))

    def get_word_count_per_chunk(self):
        """
        Returns the word count for each chunk.

        :return: List of word counts per chunk.
        """
        word_counts = [len(chunk.split()) for chunk in self.chunks]
        logger.debug("Word counts per chunk: %s", word_counts)
        return word_counts

    def get_total_chunks(self):
        """
        Returns the total number of chunks created.

        :return: Total number of chunks.
        """
        total_chunks = len(self.chunks)
        logger.info("Total chunks: %d", total_chunks)
        return total_chunks

    def get_chunks(self):
        """
        Returns the list of text chunks.

        :return: List of chunks.
        """
        return self.chunks

    def get_total_words(self):
        """
        Returns the total word count across all chunks.

        :return: Total word count.
        """
        total_words = sum(self.get_word_count_per_chunk())
        logger.info("Total word count: %d", total_words)
        return total_words


if __name__ == '__main__':
    # Example usage of ChunkManager
    text = """# Introduction to Natural Language Processing

    Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human language in a valuable way.

    ## History of NLP

    The history of NLP generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published his famous article "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence.

    The 1950s were also the decade when machine translation became a reality. The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English.

    ## Modern Approaches

    Modern NLP algorithms are based on machine learning, especially statistical machine learning. The paradigm shift was due to both the steady increase in computational power and the gradual lessening of the dominance of Chomskyan theories of linguistics.

    ### Statistical Methods

    Many different classes of machine learning algorithms have been applied to NLP tasks. These algorithms take as input a large set of "features" that are generated from the input data.

    ### Deep Learning

    In recent years, deep neural networks have achieved state-of-the-art results in many natural language processing tasks. This has led to a surge in interest in neural network approaches to NLP.

    ## Applications of NLP

    NLP is used in a wide variety of applications:

    1. Machine translation
    2. Speech recognition
    3. Sentiment analysis
    4. Question answering
    5. Information retrieval
    6. Text summarization

    Many business applications rely on NLP:
    * Customer service chatbots
    * Email filtering
    * Social media monitoring
    * Resume parsing

    ## Future Directions

    The future of NLP involves improving current methods and developing new approaches. Transfer learning has revolutionized NLP by allowing models trained on one task to be fine-tuned for another.

    Large language models like GPT, BERT, and T5 are pushing the boundaries of what's possible with NLP. These models can generate coherent paragraphs of text, answer questions, and even write creative content.

    Despite these advances, challenges remain in areas such as:
    - Common sense reasoning
    - Understanding context
    - Handling ambiguity
    - Cross-lingual transfer
    - Ethical considerations

    ## Conclusion

    NLP continues to evolve rapidly, driven by advances in machine learning, increased computing power, and the availability of large datasets. As these technologies improve, we can expect NLP systems to become more capable of understanding and generating human language in all its complexity.
    """
    chunk_manager = ChunkManager('config/config.yaml')
    
    processed_text = chunk_manager.preprocess_text(text)
    chunk_manager.flexible_chunk(processed_text)
    
    logger.info("Word count per chunk: %s", chunk_manager.get_word_count_per_chunk())
    logger.info("Total chunks: %d", chunk_manager.get_total_chunks())
    logger.info("Total words: %d", chunk_manager.get_total_words())
    logger.debug("Chunks: %s", chunk_manager.get_chunks())

    print("Chunks from textchunking.py:")
    for i, chunk in enumerate(chunk_manager.get_chunks()):
        print(f"Chunk {i+1}:\n{chunk}\n")

# #output for this:
# (brahmasumm) PS D:\brahmasumm\BrahmaSumm-Community-Edition> python .\src\chunking\textchunking.py
# INFO:__main__:Initializing ChunkManager with configuration from config/config.yaml
# INFO:__main__:ChunkManager initialized with target_words=100 and flexibility=0.25
# INFO:__main__:Preprocessing text: removing extra newlines and non-ASCII characters
# Target words 415 50 0.25
# INFO:__main__:Chunking text with target_words=50, flexibility=0.25, min_words=37, max_words=62
# INFO:__main__:Chunking completed with 9 chunks
# INFO:__main__:Word count per chunk: [48, 47, 69, 66, 37, 46, 41, 61, 0]
# INFO:__main__:Total chunks: 9
# INFO:__main__:Total chunks: 9
# INFO:__main__:Total word count: 415
# INFO:__main__:Total words: 415
# Chunks from textchunking.py:
# Chunk 1:
# # Introduction to Natural Language Processing     Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human language in a valuable way.

# Chunk 2:
#     ## History of NLP     The history of NLP generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published his famous article "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence.

# Chunk 3:
#     The 1950s were also the decade when machine translation became a reality. The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English.     ## Modern Approaches     Modern NLP algorithms are based on machine learning, especially statistical machine learning. The paradigm shift was due to both the steady increase in computational power and the gradual lessening of the dominance of Chomskyan theories of linguistics.

# Chunk 4:
#     ### Statistical Methods     Many different classes of machine learning algorithms have been applied to NLP tasks. These algorithms take as input a large set of "features" that are generated from the input data.     ### Deep Learning     In recent years, deep neural networks have achieved state-of-the-art results in many natural language processing tasks. This has led to a surge in interest in neural network approaches to NLP.

# Chunk 5:
#     ## Applications of NLP     NLP is used in a wide variety of applications:     1. Machine translation     2. Speech recognition     3. Sentiment analysis     4. Question answering     5. Information retrieval     6. Text summarization     Many business applications rely on NLP:

# Chunk 6:
#     * Customer service chatbots     * Email filtering     * Social media monitoring     * Resume parsing     ## Future Directions     The future of NLP involves improving current methods and developing new approaches. Transfer learning has revolutionized NLP by allowing models trained on one task to be fine-tuned for another.

# Chunk 7:
#     Large language models like GPT, BERT, and T5 are pushing the boundaries of what's possible with NLP. These models can generate coherent paragraphs of text, answer questions, and even write creative content.     Despite these advances, challenges remain in areas such as:

# Chunk 8:
#     - Common sense reasoning     - Understanding context     - Handling ambiguity     - Cross-lingual transfer     - Ethical considerations     ## Conclusion     NLP continues to evolve rapidly, driven by advances in machine learning, increased computing power, and the availability of large datasets. As these technologies improve, we can expect NLP systems to become more capable of understanding and generating human language in all its complexity.

# Chunk 9:

