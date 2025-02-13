import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import (
    MBartTokenizer,
    MT5Tokenizer,
    MarianTokenizer,
    AutoTokenizer
)
from sklearn.model_selection import train_test_split

import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationDataProcessor:
    def __init__(self, data_path, target_lang='ff', source_lang='en'):
        self.data_path = data_path
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.df = self._load_data()
        
    def _load_data(self):
        """Load and validate the TSV file"""
        try:
            df = pd.read_csv(self.data_path, sep='\t', names=['source', 'target'])
            # Basic data validation
            assert not df.isnull().values.any(), "NaN values found in dataset"
            assert len(df) > 100, "Dataset should have at least 100 examples"
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_path}")
            raise

    def train_val_test_split(self, test_size=0.1, val_size=0.1, random_state=42):
        """Create stratified splits maintaining sentence length distribution"""
        # Create length bins for stratification
        self.df['length_bin'] = pd.qcut(
            self.df['source'].str.len() + self.df['target'].str.len(),
            q=10,
            labels=False
        )
        
        # Stratified split
        train_df, temp_df = train_test_split(
            self.df,
            test_size=test_size + val_size,
            stratify=self.df['length_bin'],
            random_state=random_state
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size/(test_size + val_size),
            stratify=temp_df['length_bin'],
            random_state=random_state
        )
        
        return train_df.drop(columns='length_bin'), \
               val_df.drop(columns='length_bin'), \
               test_df.drop(columns='length_bin')

class ModelProcessor:
    def __init__(self, model_type, train_df, val_df, test_df):
        self.model_type = model_type
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = self._initialize_tokenizer()
        
    def _initialize_tokenizer(self):
        """Initialize appropriate tokenizer for each model type"""
        tokenizers = {
            'mbart': ('facebook/mbart-large-50', self._process_mbart),
            'mt5': ('google/mt5-base', self._process_mt5),
            'marianmt': ('Helsinki-NLP/opus-mt-en-zu', self._process_marianmt)
        }
        
        model_name, processor = tokenizers.get(self.model_type.lower(), (None, None))
        if not model_name:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Initialized {self.model_type} tokenizer from {model_name}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {str(e)}")
            raise
            
    def _process_mbart(self, text, is_target=False):
        """Process text for mBART model"""
        lang_code = 'ff_XX' if is_target else 'en_XX'
        return f'<{lang_code}> {text}'

    def _process_mt5(self, text, is_target=False):
        """Process text for mT5 model"""
        if is_target:
            return text
        return f'translate English to Fulfulde: {text}'

    def _process_marianmt(self, text, is_target=False):
        """Process text for MarianMT model"""
        return text  # Marian handles language codes internally

    def preprocess_data(self):
        """Apply model-specific preprocessing"""
        processors = {
            'source': {
                'mbart': self._process_mbart,
                'mt5': self._process_mt5,
                'marianmt': self._process_marianmt
            },
            'target': {
                'mbart': lambda x: self._process_mbart(x, is_target=True),
                'mt5': lambda x: x,
                'marianmt': lambda x: x
            }
        }
        
        for df in [self.train_df, self.val_df, self.test_df]:
            df['source'] = df['source'].apply(processors['source'][self.model_type])
            df['target'] = df['target'].apply(processors['target'][self.model_type])
            
        return self.train_df, self.val_df, self.test_df

    def tokenize_dataset(self, dataset):
        """Tokenize dataset with model-specific handling"""
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['source'],
                text_target=examples['target'],
                max_length=128,
                truncation=True,
                padding='max_length',
                return_tensors='np'
            )
            return tokenized
        
        return dataset.map(tokenize_function, batched=True)

    def analyze_tokenization(self, tokenized_dataset):
        """Generate quality report for tokenization"""
        report = {
            'source_lengths': [],
            'target_lengths': [],
            'unk_tokens': []
        }
        
        for example in tokenized_dataset:
            report['source_lengths'].append(len(example['input_ids']))
            report['target_lengths'].append(len(example['labels']))
            report['unk_tokens'].append(
                (np.array(example['input_ids']) == self.tokenizer.unk_token_id).sum()
            )
            
        logger.info("\nTokenization Quality Report:")
        logger.info(f"Average source length: {np.mean(report['source_lengths']):.1f}")
        logger.info(f"Average target length: {np.mean(report['target_lengths']):.1f}")
        logger.info(f"UNK tokens per sample: {np.mean(report['unk_tokens']):.2f}")
        
        return report

def main():
    DATA_PATH = "./data/adamawa_english_fulfulde_french.tsv"
    
    try:
        # Initialize data processor
        data_processor = TranslationDataProcessor(DATA_PATH)
        train_df, val_df, test_df = data_processor.train_val_test_split()
        
        # Example processing for all model types
        # for model_type in ['mbart', 'mt5', 'marianmt']:
        for model_type in ['mbart', 'mt5', 'marianmt']:
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing for {model_type.upper()}")
            logger.info(f"{'='*40}")
            
            # Initialize model processor
            processor = ModelProcessor(
                model_type=model_type,
                train_df=train_df.copy(),
                val_df=val_df.copy(),
                test_df=test_df.copy()
            )
            
            # Preprocess data
            train_proc, val_proc, test_proc = processor.preprocess_data()
            
            # Convert to Dataset
            train_ds = Dataset.from_pandas(train_proc)
            val_ds = Dataset.from_pandas(val_proc)
            test_ds = Dataset.from_pandas(test_proc)
            
            # Tokenize datasets
            tokenized_train = processor.tokenize_dataset(train_ds)
            tokenized_val = processor.tokenize_dataset(val_ds)
            tokenized_test = processor.tokenize_dataset(test_ds)
            
            # Generate quality report
            _ = processor.analyze_tokenization(tokenized_train)
            
            # Save datasets
            save_path = f"processed_data/{model_type}"
            tokenized_train.save_to_disk(f"{save_path}/train")
            tokenized_val.save_to_disk(f"{save_path}/val")
            tokenized_test.save_to_disk(f"{save_path}/test")
            
            logger.info(f"Sample tokenized input: {processor.tokenizer.decode(tokenized_train[0]['input_ids'])}")
            logger.info(f"Sample tokenized target: {processor.tokenizer.decode(tokenized_train[0]['labels'])}")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
