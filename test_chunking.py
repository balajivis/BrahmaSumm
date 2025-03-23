import sys
import os
import logging
from src.chunking.textchunking import ChunkManager

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_chunking(sample_text, config_path='config/config.yaml'):
    """Test the current chunking strategy on a sample text."""
    logger.info("Testing chunking with text length: %d characters", len(sample_text))
    
    # Initialize the chunk manager
    chunk_manager = ChunkManager(config_path)
    
    # Preprocess the text
    logger.info("Preprocessing text...")
    processed_text = chunk_manager.preprocess_text(sample_text)
    
    # Chunk the text
    logger.info("Chunking text...")
    chunk_manager.flexible_chunk(processed_text)
    
    # Get and display results
    chunks = chunk_manager.get_chunks()
    word_counts = chunk_manager.get_word_count_per_chunk()
    total_chunks = chunk_manager.get_total_chunks()
    total_words = chunk_manager.get_total_words()
    
    # Display chunking results
    logger.info("Chunking Results:")
    logger.info("Total chunks: %d", total_chunks)
    logger.info("Total words: %d", total_words)
    logger.info("Word counts per chunk: %s", word_counts)
    
    # Display each chunk with its word count
    logger.info("\nDetailed Chunk Analysis:")
    for i, (chunk, word_count) in enumerate(zip(chunks, word_counts)):
        logger.info(f"Chunk {i+1}/{total_chunks} - {word_count} words:")
        # Display first 100 characters of each chunk
        preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
        logger.info(f"Preview: {preview}")
        logger.info("-" * 50)
    
    return chunks, word_counts

def main():
    # Sample text for testing - a mix of paragraphs, headers, and different structures
    sample_text = """
    # Introduction to Natural Language Processing

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
    
    # Run the test
    chunks, word_counts = test_chunking(sample_text)
    
    # Print statistics
    print("\nSummary Statistics:")
    print(f"Average chunk size: {sum(word_counts)/len(word_counts):.2f} words")
    print(f"Minimum chunk size: {min(word_counts)} words")
    print(f"Maximum chunk size: {max(word_counts)} words")
    print(f"Standard deviation: {(sum((x - sum(word_counts)/len(word_counts))**2 for x in word_counts) / len(word_counts))**0.5:.2f} words")

if __name__ == "__main__":
    main()
