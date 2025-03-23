import re
import yaml
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK packages are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkManager:
    """
    Enhanced ChunkManager class responsible for intelligently dividing documents into chunks
    based on content semantics, document structure, and configurable parameters.
    It preserves document hierarchy, enables overlap for context continuity, and adapts
    to different document types and lengths.
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
        self.overlap = self.config.get('chunk_overlap', 0.1)           # Default to 10% overlap between chunks
        self.min_chunk_size = self.config.get('min_chunk_size', 40)    # Minimum words in a chunk
        self.respect_headers = self.config.get('respect_headers', True) # Respect document headers in chunking
        self.semantic_chunking = self.config.get('semantic_chunking', True) # Use semantic similarity for chunking
        self.chunks = []
        self.chunk_metadata = []  # Store metadata about chunks (headers, section info, etc.)
        
        logger.info(f"ChunkManager initialized with target_words={self.target_words}, "
                   f"flexibility={self.flexibility}, overlap={self.overlap}")

    def preprocess_text(self, text: str, doc_type: Optional[str] = None) -> str:
        """
        Enhanced preprocessing that cleans and normalizes text based on document type.
        
        :param text: The raw text to be preprocessed.
        :param doc_type: Optional document type (pdf, webpage, word, etc.) for type-specific processing.
        :return: Cleaned and normalized text.
        """
        logger.info(f"Preprocessing text for document type: {doc_type or 'unknown'}")
        
        # Common preprocessing for all document types
        text = re.sub(r'\n+', '\n\n', text)  # Normalize multiple newlines
        text = re.sub(r'\t+', '\t', text)    # Normalize multiple tabs
        
        # Handle non-ASCII characters more intelligently (keep useful Unicode)
        text = re.sub(r'[^\x00-\x7F\u2013\u2014\u2018\u2019\u201C\u201D\u2022\u2026\u00A0-\u00FF]+', ' ', text)
        
        # Document type-specific preprocessing
        if doc_type == 'pdf':
            # PDF-specific cleaning (handle page numbers, headers/footers)
            text = re.sub(r'\n\s*\d+\s*\n', '\n\n', text)  # Remove isolated page numbers
            text = re.sub(r'(\n\s*){3,}', '\n\n', text)    # Normalize excessive whitespace between PDF sections
            
        elif doc_type == 'webpage':
            # Webpage-specific cleaning (handle HTML artifacts)
            text = re.sub(r'\s*\[\d+\]\s*', ' ', text)     # Remove reference numbers like [1], [2]
            text = re.sub(r'\s{2,}', ' ', text)            # Normalize excessive spaces common in HTML
            
        elif doc_type == 'word':
            # Word document specific cleaning
            text = re.sub(r'_+', '', text)                 # Remove underscores used for formatting
            
        # Detect and preserve headers/section titles
        if self.respect_headers:
            # Match potential headers (all caps lines, numbered sections, etc.)
            text = re.sub(r'(\n\s*)([A-Z][A-Z\s]+:?|\d+\.\s+.+?)(\n)', r'\1##HEADER##\2##HEADER##\3', text)
            
        logger.debug("Preprocessed text (first 100 chars): %s", text[:100])
        return text

    def _detect_document_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyzes document to detect its structure - headers, sections, hierarchies.
        
        :param text: Preprocessed text
        :return: Dictionary containing document structure information
        """
        structure = {
            'headers': [],
            'sections': [],
            'avg_sentence_length': 0,
            'avg_paragraph_length': 0,
            'has_numbered_sections': False
        }
        
        # Detect headers (marked during preprocessing)
        headers = re.findall(r'##HEADER##(.*?)##HEADER##', text)
        structure['headers'] = headers
        
        # Check for numbered sections
        numbered_sections = re.findall(r'\n\s*\d+\.\d*\s+[A-Z]', text)
        structure['has_numbered_sections'] = len(numbered_sections) > 3  # Heuristic: if >3 matches, likely has numbered sections
        
        # Calculate average sentence length
        sentences = sent_tokenize(text)
        structure['avg_sentence_length'] = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        # Calculate average paragraph length
        paragraphs = re.split(r'\n\n', text)
        structure['avg_paragraph_length'] = sum(len(p.split()) for p in paragraphs) / max(1, len(paragraphs))
        
        # Store section information
        current_section = ""
        sections = []
        
        for line in text.split('\n'):
            if '##HEADER##' in line:
                if current_section:
                    sections.append(current_section)
                current_section = re.sub(r'##HEADER##', '', line).strip()
            elif current_section:
                current_section += f"\n{line}"
                
        if current_section:
            sections.append(current_section)
            
        structure['sections'] = sections
        
        logger.info(f"Document structure analysis: {len(headers)} headers, {len(sections)} sections, "
                   f"avg_sentence_length={structure['avg_sentence_length']:.1f} words")
        
        return structure

    def _calculate_adaptive_parameters(self, text: str, doc_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates adaptive chunking parameters based on document characteristics.
        
        :param text: Preprocessed text
        :param doc_structure: Document structure information
        :return: Dictionary of adaptive chunking parameters
        """
        word_count = len(text.split())
        params = {}
        
        # Adapt target words based on document length
        if word_count < 500:
            params['target_words'] = min(50, self.target_words)
            params['flexibility'] = min(0.4, self.flexibility + 0.15)  # More flexibility for short docs
            
        elif word_count < 2000:
            params['target_words'] = min(75, self.target_words)
            params['flexibility'] = self.flexibility
            
        elif word_count < 10000:
            params['target_words'] = self.target_words
            params['flexibility'] = self.flexibility
            
        else:  # Very long documents
            params['target_words'] = max(self.target_words, 150)  # Larger chunks for very long docs
            params['flexibility'] = max(0.15, self.flexibility - 0.1)  # Less flexibility for long docs
            
        # Adapt based on document structure
        if doc_structure['avg_sentence_length'] > 25:  # Document has long sentences
            params['target_words'] = max(params.get('target_words', self.target_words), 120)
            
        if doc_structure['avg_paragraph_length'] < 40:  # Document has short paragraphs
            params['respect_paragraphs'] = True
        else:
            params['respect_paragraphs'] = False
            
        if doc_structure['has_numbered_sections']:
            params['section_aware'] = True
            
        logger.info(f"Adaptive parameters calculated: target_words={params.get('target_words')}, "
                   f"flexibility={params.get('flexibility')}")
        
        return params

    def _semantic_similarity(self, text_chunks: List[str]) -> np.ndarray:
        """
        Calculate semantic similarity between text chunks using TF-IDF and cosine similarity.
        
        :param text_chunks: List of text chunks to analyze
        :return: Similarity matrix between chunks
        """
        if len(text_chunks) < 2:
            return np.array([[1.0]])
            
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(text_chunks)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            # Return identity matrix as fallback
            return np.eye(len(text_chunks))

    def _merge_similar_chunks(self, chunks: List[str], similarity_threshold: float = 0.7) -> List[str]:
        """
        Merge chunks that are semantically very similar to create more coherent chunks.
        
        :param chunks: List of initial chunks
        :param similarity_threshold: Threshold for merging chunks
        :return: List of merged chunks
        """
        if len(chunks) < 2:
            return chunks
            
        sim_matrix = self._semantic_similarity(chunks)
        merged_chunks = []
        skip_indices = set()
        
        for i in range(len(chunks)):
            if i in skip_indices:
                continue
                
            current_chunk = chunks[i]
            
            # Check similarity with next chunk
            if i+1 < len(chunks) and sim_matrix[i, i+1] > similarity_threshold:
                combined_chunk = f"{current_chunk} {chunks[i+1]}"
                merged_chunks.append(combined_chunk)
                skip_indices.add(i+1)
            else:
                merged_chunks.append(current_chunk)
                
        logger.info(f"Merged {len(chunks) - len(merged_chunks)} similar chunks")
        return merged_chunks

    def flexible_chunk(self, text: str, doc_type: Optional[str] = None, target_words: Optional[int] = None, 
                      flexibility: Optional[float] = None) -> None:
        """
        Enhanced chunking algorithm that intelligently divides text based on semantics, structure,
        and configurable parameters with adaptive behavior based on document characteristics.

        :param text: The preprocessed text to be chunked
        :param doc_type: Optional document type for type-specific processing
        :param target_words: Optional target word count per chunk
        :param flexibility: Optional flexibility percentage for chunk size
        """
        # Clean preprocessing markers that should not appear in final chunks
        clean_text = re.sub(r'##HEADER##', '', text)
        
        # Analyze document structure
        doc_structure = self._detect_document_structure(text)
        
        # Calculate adaptive parameters
        adaptive_params = self._calculate_adaptive_parameters(clean_text, doc_structure)
        
        # Use provided parameters or fall back to adaptive/default
        actual_target_words = target_words or adaptive_params.get('target_words', self.target_words)
        actual_flexibility = flexibility or adaptive_params.get('flexibility', self.flexibility)
        respect_paragraphs = adaptive_params.get('respect_paragraphs', True)
        section_aware = adaptive_params.get('section_aware', False)
        
        # Calculate min/max words boundaries
        min_words = max(self.min_chunk_size, int(actual_target_words * (1 - actual_flexibility)))
        max_words = int(actual_target_words * (1 + actual_flexibility))
        
        logger.info(f"Chunking with target_words={actual_target_words}, flexibility={actual_flexibility}, "
                   f"min_words={min_words}, max_words={max_words}, section_aware={section_aware}")
        
        # Split text into initial segments (paragraphs or sections)
        if section_aware and doc_structure['sections']:
            initial_segments = doc_structure['sections']
            logger.info(f"Using {len(initial_segments)} document sections as initial segments")
        else:
            initial_segments = re.split(r'\n\n', clean_text)
            logger.info(f"Using {len(initial_segments)} paragraphs as initial segments")
        
        chunks = []
        chunk_metadata = []
        current_chunk = []
        current_word_count = 0
        current_header = None
        
        # Track headers for metadata
        header_pattern = r'^\s*([A-Z][A-Z0-9\s]+:?|\d+\.\d*\s+.+?)$'
        
        # Function to finalize a chunk
        def finalize_chunk(force: bool = False) -> None:
            nonlocal current_chunk, current_word_count, current_header
            if current_chunk and (force or current_word_count >= min_words):
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Store metadata
                metadata = {
                    'word_count': current_word_count,
                    'header': current_header,
                    'starts_with_header': current_chunk[0].strip().upper() == current_chunk[0].strip(),
                }
                chunk_metadata.append(metadata)
                
                # Handle overlap for next chunk if not forced
                if not force and self.overlap > 0 and len(current_chunk) > 2:
                    # Calculate how many sentences to keep for overlap
                    overlap_sentences = current_chunk[-2:]  # Default to last 2 items
                    overlap_word_count = sum(len(s.split()) for s in overlap_sentences)
                    
                    # Reset for next chunk but keep overlap
                    current_chunk = overlap_sentences
                    current_word_count = overlap_word_count
                else:
                    # Reset for next chunk with no overlap
                    current_chunk = []
                    current_word_count = 0
        
        # Process each segment and create chunks
        for segment in initial_segments:
            segment = segment.strip()
            if not segment:
                continue
                
            # Check if segment starts with a header
            header_match = re.match(header_pattern, segment.split('\n')[0], re.MULTILINE)
            if header_match:
                current_header = header_match.group(1)
            
            # Check if segment is short enough to be a chunk on its own
            segment_word_count = len(segment.split())
            
            if segment_word_count <= max_words:
                # If adding this segment keeps us under max, add it
                if current_word_count + segment_word_count <= max_words:
                    current_chunk.append(segment)
                    current_word_count += segment_word_count
                else:
                    # Finalize current chunk and start new one with this segment
                    finalize_chunk()
                    current_chunk = [segment]
                    current_word_count = segment_word_count
                    
                # If we've reached minimum size, check if we should finalize
                if current_word_count >= min_words:
                    finalize_chunk()
            else:
                # Segment is too large, process it sentence by sentence
                if current_chunk:
                    finalize_chunk()
                
                sentences = sent_tokenize(segment)
                current_chunk = []
                current_word_count = 0
                
                for sentence in sentences:
                    sentence_word_count = len(sentence.split())
                    
                    # If sentence itself exceeds max words, split it further
                    if sentence_word_count > max_words:
                        if current_chunk:
                            finalize_chunk()
                            
                        # Split long sentence by phrases (using commas, etc.)
                        phrases = re.split(r'[,;:]', sentence)
                        current_chunk = []
                        current_word_count = 0
                        
                        for phrase in phrases:
                            phrase = phrase.strip()
                            if not phrase:
                                continue
                                
                            phrase_word_count = len(phrase.split())
                            if current_word_count + phrase_word_count <= max_words:
                                current_chunk.append(phrase)
                                current_word_count += phrase_word_count
                            else:
                                finalize_chunk()
                                current_chunk = [phrase]
                                current_word_count = phrase_word_count
                    else:
                        # Normal sentence handling
                        if current_word_count + sentence_word_count <= max_words:
                            current_chunk.append(sentence)
                            current_word_count += sentence_word_count
                        else:
                            finalize_chunk()
                            current_chunk = [sentence]
                            current_word_count = sentence_word_count
                
                # Finalize any remaining content
                if current_chunk:
                    finalize_chunk()
        
        # Finalize any remaining content at the end
        finalize_chunk(force=True)
        
        # Perform semantic-based optimizations if enabled
        if self.semantic_chunking and len(chunks) > 1:
            chunks = self._merge_similar_chunks(chunks)
        
        self.chunks = chunks
        self.chunk_metadata = chunk_metadata
        logger.info(f"Chunking completed with {len(chunks)} chunks")

    def get_word_count_per_chunk(self) -> List[int]:
        """
        Returns the word count for each chunk.

        :return: List of word counts per chunk.
        """
        word_counts = [len(chunk.split()) for chunk in self.chunks]
        logger.debug("Word counts per chunk: %s", word_counts)
        return word_counts

    def get_total_chunks(self) -> int:
        """
        Returns the total number of chunks created.

        :return: Total number of chunks.
        """
        total_chunks = len(self.chunks)
        logger.info("Total chunks: %d", total_chunks)
        return total_chunks

    def get_chunks(self) -> List[str]:
        """
        Returns the list of text chunks.

        :return: List of chunks.
        """
        return self.chunks
        
    def get_chunk_metadata(self) -> List[Dict[str, Any]]:
        """
        Returns metadata about the chunks.
        
        :return: List of chunk metadata dictionaries.
        """
        return self.chunk_metadata

    def get_total_words(self) -> int:
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
    
    processed_text = chunk_manager.preprocess_text(text, doc_type='pdf')
    chunk_manager.flexible_chunk(processed_text, doc_type='pdf')
    
    logger.info("Word count per chunk: %s", chunk_manager.get_word_count_per_chunk())
    logger.info("Total chunks: %d", chunk_manager.get_total_chunks())
    logger.info("Total words: %d", chunk_manager.get_total_words())
    logger.debug("Chunks: %s", chunk_manager.get_chunks())
    logger.debug("Chunk metadata: %s", chunk_manager.get_chunk_metadata())

    print("Chunks from textchunking_v2.py:")
    for i, chunk in enumerate(chunk_manager.get_chunks()):
        print(f"Chunk {i+1}:\n{chunk}\n")

#the output for this was:
# (brahmasumm) PS D:\brahmasumm\BrahmaSumm-Community-Edition> python .\src\chunking\textchunking_v2.py
# INFO:__main__:Initializing ChunkManager with configuration from config/config.yaml
# INFO:__main__:ChunkManager initialized with target_words=100, flexibility=0.25, overlap=0.1
# INFO:__main__:Preprocessing text for document type: pdf
# INFO:__main__:Document structure analysis: 6 headers, 6 sections, avg_sentence_length=17.3 words
# INFO:__main__:Adaptive parameters calculated: target_words=50, flexibility=0.4
# INFO:__main__:Chunking with target_words=50, flexibility=0.4, min_words=40, max_words=70, section_aware=False
# INFO:__main__:Using 36 paragraphs as initial segments
# INFO:__main__:Merged 3 similar chunks
# INFO:__main__:Chunking completed with 11 chunks
# INFO:__main__:Word count per chunk: [48, 47, 69, 113, 66, 46, 41, 52, 134, 45, 115]
# INFO:__main__:Total chunks: 11
# INFO:__main__:Total chunks: 11
# INFO:__main__:Total word count: 776
# INFO:__main__:Total words: 776
# Chunks from textchunking_v2.py:
# Chunk 1:
# # Introduction to Natural Language Processing Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human language in a valuable way.

# Chunk 2:
# ## History of NLP The history of NLP generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published his famous article "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence.

# Chunk 3:
# The 1950s were also the decade when machine translation became a reality. The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. ## Modern Approaches Modern NLP algorithms are based on machine learning, especially statistical machine learning. The paradigm shift was due to both the steady increase in computational power and the gradual lessening of the dominance of Chomskyan theories of linguistics.

# Chunk 4:
# ## Modern Approaches Modern NLP algorithms are based on machine learning, especially statistical machine learning. The paradigm shift was due to both the steady increase in computational power and the gradual lessening of the dominance of Chomskyan theories of linguistics. ### Statistical Methods Modern NLP algorithms are based on machine learning, especially statistical machine learning. The paradigm shift was due to both the steady increase in computational power and the gradual lessening of the dominance of Chomskyan theories of linguistics. ### Statistical Methods Many different classes of machine learning algorithms have been applied to NLP tasks. These algorithms take as input a large set of "features" that are generated from the input data.

# Chunk 5:
# ### Statistical Methods Many different classes of machine learning algorithms have been applied to NLP tasks. These algorithms take as input a large set of "features" that are generated from the input data. ### Deep Learning In recent years, deep neural networks have achieved state-of-the-art results in many natural language processing tasks. This has led to a surge in interest in neural network approaches to NLP.       

# Chunk 6:
# ### Deep Learning In recent years, deep neural networks have achieved state-of-the-art results in many natural language processing tasks. This has led to a surge in interest in neural network approaches to NLP. ## Applications of NLP NLP is used in a wide variety of applications:

# Chunk 7:
# ## Applications of NLP NLP is used in a wide variety of applications: 1. Machine translation 2. Speech recognition 3. Sentiment analysis 4. Question answering 5. Information retrieval 6. Text summarization Many business applications rely on NLP: * Customer service chatbots

# Chunk 8:
# Many business applications rely on NLP: * Customer service chatbots * Email filtering * Social media monitoring * Resume parsing ## Future Directions The future of NLP involves improving current methods and developing new approaches. Transfer learning has revolutionized NLP by allowing models trained on one task to be fine-tuned for another.

# Chunk 9:
# ## Future Directions The future of NLP involves improving current methods and developing new approaches. Transfer learning has revolutionized NLP by allowing models trained on one task to be fine-tuned for another. Large language models like GPT, BERT, and T5 are pushing the boundaries of what's possible with NLP. These models can generate coherent paragraphs of text, answer questions, and even write creative content. The future of NLP involves improving current methods and developing new approaches. Transfer learning has revolutionized NLP by allowing models trained on one task to be fine-tuned for another. Large language models like GPT, BERT, and T5 are pushing the boundaries of what's possible with NLP. These models can generate coherent paragraphs of text, answer questions, and even write creative content. Despite these advances, challenges remain in areas such as:

# Chunk 10:
# Large language models like GPT, BERT, and T5 are pushing the boundaries of what's possible with NLP. These models can generate coherent paragraphs of text, answer questions, and even write creative content. Despite these advances, challenges remain in areas such as: - Common sense reasoning

# Chunk 11:
# Despite these advances, challenges remain in areas such as: - Common sense reasoning - Understanding context - Handling ambiguity - Cross-lingual transfer - Ethical considerations ## Conclusion NLP continues to evolve rapidly, driven by advances in machine learning, increased computing power, and the availability of large datasets. As these technologies improve, we can expect NLP systems to become more capable of understanding and generating human language in all its complexity. ## Conclusion NLP continues to evolve rapidly, driven by advances in machine learning, increased computing power, and the availability of large datasets. As these technologies improve, we can expect NLP systems to become more capable of understanding and generating human language in all its complexity.