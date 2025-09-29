import pandas as pd
import os
import re
from typing import List, Tuple
from datetime import datetime

class PunctuationContextBuilder:
    """
    Build context based on sentence completion markers (., !, ?, :, ;)
    Groups quasi-sentences into complete thoughts using punctuation
    """
    
    def __init__(self, max_context_sentences=10, include_previous_complete=True):
        self.max_context_sentences = max_context_sentences
        self.include_previous_complete = include_previous_complete
        
        # Sentence-ending punctuation
        self.sentence_enders = r'[.!?]'
        # Clause-ending punctuation (weaker boundaries)
        self.clause_enders = r'[.!?:;]'
    
    def analyze_punctuation_structure(self, df: pd.DataFrame, text_column='text_en') -> pd.DataFrame:
        """Analyze punctuation patterns in the data"""
        
        df = df.copy()
        
        def get_punctuation_info(text):
            text = str(text).strip()
            
            return {
                'ends_with_period': text.endswith('.'),
                'ends_with_question': text.endswith('?'),
                'ends_with_exclamation': text.endswith('!'),
                'ends_with_colon': text.endswith(':'),
                'ends_with_semicolon': text.endswith(';'),
                'ends_with_comma': text.endswith(','),
                'is_sentence_complete': bool(re.search(self.sentence_enders + r'$', text)),
                'is_clause_complete': bool(re.search(self.clause_enders + r'$', text)),
                'has_no_punctuation': not bool(re.search(r'[.!?:;,]$', text)),
                'text_length': len(text),
                'word_count': len(text.split())
            }
        
        # Apply analysis
        punct_info = df[text_column].apply(get_punctuation_info)
        punct_df = pd.DataFrame(punct_info.tolist())
        
        # Combine with original data
        result_df = pd.concat([df, punct_df], axis=1)
        
        return result_df
    
    def find_complete_thoughts(self, df: pd.DataFrame, text_column='text_en') -> List[List[int]]:
        """
        Group quasi-sentences into complete thoughts based on punctuation
        
        Returns:
            List of lists, where each inner list contains indices of sentences 
            that form one complete thought
        """
        
        complete_thoughts = []
        current_thought = []
        
        for i, row in df.iterrows():
            current_thought.append(i)
            
            # Check if this sentence completes a thought
            text = str(row[text_column]).strip()
            
            # End thought if:
            # 1. Sentence ends with strong punctuation (. ! ?)
            # 2. We've reached maximum length
            # 3. This is the last sentence
            
            ends_thought = False
            
            if re.search(self.sentence_enders + r'$', text):
                ends_thought = True
            elif len(current_thought) >= self.max_context_sentences:
                ends_thought = True
            elif i == len(df) - 1:  # Last sentence
                ends_thought = True
            
            if ends_thought:
                complete_thoughts.append(current_thought.copy())
                current_thought = []
        
        # Handle any remaining incomplete thought
        if current_thought:
            complete_thoughts.append(current_thought)
        
        return complete_thoughts
    
    def build_contexts_from_thoughts(self, df: pd.DataFrame, text_column='text_en') -> pd.DataFrame:
        """
        Build context for each sentence based on complete thoughts
        """
        
        df = df.copy()
        
        # Find complete thoughts
        complete_thoughts = self.find_complete_thoughts(df, text_column)
        
        print(f"Found {len(complete_thoughts)} complete thoughts")
        
        # Build context for each sentence
        contexts = [''] * len(df)
        thought_ids = [0] * len(df)
        
        for thought_id, thought_indices in enumerate(complete_thoughts):
            
            # Get text for this complete thought
            thought_sentences = [df.iloc[idx][text_column] for idx in thought_indices]
            current_thought_text = ' '.join(thought_sentences)
            
            # Optionally include previous complete thought for more context
            context_parts = []
            
            if self.include_previous_complete and thought_id > 0:
                # Add previous complete thought
                prev_thought_indices = complete_thoughts[thought_id - 1]
                prev_sentences = [df.iloc[idx][text_column] for idx in prev_thought_indices]
                prev_thought_text = ' '.join(prev_sentences)
                context_parts.append(prev_thought_text)
            
            # Add current complete thought
            context_parts.append(current_thought_text)
            
            # Assign this context to all sentences in the thought
            full_context = ' '.join(context_parts)
            
            for idx in thought_indices:
                contexts[idx] = full_context
                thought_ids[idx] = thought_id
        
        df['context'] = contexts
        df['thought_id'] = thought_ids
        
        return df
    
    def validate_and_report(self, df: pd.DataFrame, text_column='text_en'):
        """Generate validation report"""
        
        print("=== Punctuation-Based Context Analysis ===")
        
        # Punctuation statistics
        print("\nPunctuation Distribution:")
        print(f"Sentences ending with period: {df['ends_with_period'].sum()}")
        print(f"Sentences ending with question mark: {df['ends_with_question'].sum()}")
        print(f"Sentences ending with exclamation: {df['ends_with_exclamation'].sum()}")
        print(f"Sentences with no punctuation: {df['has_no_punctuation'].sum()}")
        print(f"Complete sentences: {df['is_sentence_complete'].sum()}")
        
        # Thought distribution
        thought_sizes = df.groupby('thought_id').size()
        print(f"\nComplete Thoughts Analysis:")
        print(f"Number of complete thoughts: {len(thought_sizes)}")
        print(f"Average sentences per thought: {thought_sizes.mean():.1f}")
        print(f"Thought size distribution: {thought_sizes.value_counts().sort_index().to_dict()}")
        
        # Context quality
        unique_contexts = df['context'].nunique()
        print(f"\nContext Quality:")
        print(f"Unique contexts: {unique_contexts}/{len(df)}")
        print(f"Average context length: {df['context'].str.len().mean():.0f} characters")
        
        # Show examples
        print(f"\n=== Examples ===")
        for thought_id in range(min(3, df['thought_id'].max() + 1)):
            thought_sentences = df[df['thought_id'] == thought_id]
            print(f"\nComplete Thought {thought_id}:")
            
            for idx, row in thought_sentences.iterrows():
                print(f"  Sentence {idx}: {row[text_column]}")
            
            print(f"  Context: {thought_sentences.iloc[0]['context'][:200]}...")
            print("-" * 60)

def test_punctuation_builder():
    """Test with sample data"""
    
    # Sample data that shows punctuation patterns
    sample_data = pd.DataFrame({
        'text_en': [
            "We can all feel it: things cannot stay as they are.",           # Complete with period
            "Because we cannot be satisfied with the fact that jobs",       # Incomplete fragment  
            "and prosperity are at risk.",                                   # Completes previous
            "That parents and grandparents are worried about",              # New incomplete
            "their children's educational opportunities?",                   # Question mark
            "That internal",                                                 # Fragment
            "and external security are more challenged than ever.",          # Completes with period
            "What can we do about this situation?",                         # Complete question
            "We must act now!"                                              # Complete exclamation
        ],
        'cmp_code': ["000", "701", "410", "506", "506", "605.1", "104", "104", "104"]
    })
    
    print("=== Testing Punctuation-Based Context Builder ===")
    
    # Initialize builder
    builder = PunctuationContextBuilder(
        max_context_sentences=10,
        include_previous_complete=True
    )
    
    # Analyze punctuation structure
    df_analyzed = builder.analyze_punctuation_structure(sample_data)
    
    # Build contexts
    df_with_context = builder.build_contexts_from_thoughts(df_analyzed)
    
    # Generate report
    builder.validate_and_report(df_with_context)
    
    return df_with_context

def create_sentence_pairs(df, text_column='text_en', max_context_chars=1000):
    """Create final sentence pairs for training"""
    
    df = df.copy()
    
    def create_pair(row):
        target = str(row[text_column]).strip()
        context = str(row['context']).strip()
        
        # Truncate if needed
        if len(context) > max_context_chars:
            context = context[:max_context_chars]
        
        return f"{target} </s> </s> {context} </s>"
    
    df['input_text'] = df.apply(create_pair, axis=1)
    return df

def process_csv_with_punctuation(csv_path):
    """Process your actual CSV with punctuation-based context"""
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} sentences from {csv_path}")
        
        # Build contexts
        builder = PunctuationContextBuilder(
            max_context_sentences=8,  # Allow longer thoughts
            include_previous_complete=True
        )
        
        # Analyze and build contexts
        df_analyzed = builder.analyze_punctuation_structure(df)
        df_with_context = builder.build_contexts_from_thoughts(df_analyzed)
        
        # Create training pairs
        df_final = create_sentence_pairs(df_with_context)
        
        # Validation report
        builder.validate_and_report(df_with_context)
        
        return df_final
        
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    
    # Fixed paths and variables
    paths = ["data/afd.csv", "data/spd.csv", "data/cdu.csv", "data/fdp.csv", "data/gruene.csv", "data/linke.csv", "data/bsw.csv", "data/sswb.csv"]
    DAYTIME = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main output directory
    main_output_dir = f"data/contextaware_data_{DAYTIME}"
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
    
    print("=" * 80)
    print("Processing actual data...")
    
    # Process each file
    for path in paths:
        print(f"\n--- Processing {path} ---")
        
        # Check if input file exists
        if not os.path.exists(path):
            print(f"Warning: File {path} not found, skipping...")
            continue
        
        # Process the CSV
        result = process_csv_with_punctuation(path)  # Fixed: was csv_path, now path
        
        if result is not None:
            # Extract party name from filename
            party_name = os.path.splitext(os.path.basename(path))[0]
            
            # Create output filename
            output_filename = f"{party_name}_context_{DAYTIME}.csv"
            output_path = os.path.join(main_output_dir, output_filename)
            
            # Save the result
            result.to_csv(output_path, index=False)
            print(f"Saved to: {output_path}")
            
            # Show some examples
            print(f"\nFirst 2 examples from {party_name}:")
            for i in range(min(2, len(result))):
                print(f"\nExample {i+1}:")
                print(f"Target: {result.iloc[i]['text_en']}")
                print(f"Label: {result.iloc[i]['cmp_code']}")
                print(f"Input: {result.iloc[i]['input_text'][:150]}...")
        else:
            print(f"Failed to process {path}")
    
    print(f"\n" + "=" * 80)
    print(f"All files processed. Results saved in: {main_output_dir}")