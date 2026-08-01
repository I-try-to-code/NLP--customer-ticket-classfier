import sys
import re
import contractions
from src.exception import CustomException
from src.utils.tokenizer import TextTokenizer
from src.utils.stopword_remover import StopwordRemover
from src.utils.lemmatizer import Lemmatizer

class NLPPreprocessingPipeline:
    def __init__(self):
        """
        Initialize the NLP preprocessing pipeline components.
        """
        self.stopword_remover = StopwordRemover()
        self.lemmatizer = Lemmatizer()
        
    def _clean_text(self, text: str) -> str:
        """
        Applies initial text cleaning like regex substitutions.
        """
        if not isinstance(text, str):
            return text
            
        # 1. Lowercase
        text = text.lower()
        # 2. Remove HTML
        text = re.sub(r'<.*?>', '', text)
        # 3. Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # 4. Remove Emails
        text = re.sub(r'\S+@\S+', '', text)
        # 5. Expand Contractions
        text = contractions.fix(text)
        # 6. Replace literal newline/tab characters
        text = text.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
        # 7. Normalize Whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def transform(self, raw_text: str) -> str:
        """
        Executes the full NLP preprocessing pipeline:
        Cleaning -> Tokenization -> Stopword Removal -> Lemmatization -> Join Tokens
        """
        try:
            if not isinstance(raw_text, str):
                return raw_text
                
            # Step 1: Cleaning
            cleaned_text = self._clean_text(raw_text)
            
            # Step 2: Tokenization
            tokens = TextTokenizer.tokenize(cleaned_text)
            
            # Step 3: Stopword Removal
            filtered_tokens = self.stopword_remover.remove_stopwords(tokens)
            
            # Step 4: Lemmatization
            lemmatized_tokens = self.lemmatizer.lemmatize(filtered_tokens)
            
            # Step 5: Join Tokens
            final_text = " ".join(lemmatized_tokens)
            
            return final_text
            
        except Exception as e:
            raise CustomException(e, sys)
