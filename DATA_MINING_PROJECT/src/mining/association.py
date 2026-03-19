# ==============================================================================
# ASSOCIATION RULES MINING MODULE
# ==============================================================================
"""
Module khai phá luật kết hợp (Association Rules Mining):
- Phân tích luật kết hợp trên từ khoá / khía cạnh (aspect)
- Tìm top luật dịch vụ đi kèm phàn nàn / khen
- Báo cáo: support, confidence, lift
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


class AspectEncoder:
    """Mã hoá aspects thành transaction format."""
    
    def __init__(self, keywords_dict: Optional[Dict[str, List[str]]] = None):
        self.keywords_dict = keywords_dict or {
            'room': ['room', 'bed', 'bathroom', 'towel', 'shower', 'clean', 'comfortable'],
            'service': ['service', 'staff', 'friendly', 'helpful', 'professional', 'rude'],
            'location': ['location', 'central', 'near', 'close', 'convenient', 'far'],
            'food': ['food', 'breakfast', 'dinner', 'restaurant', 'buffet', 'delicious'],
            'price': ['price', 'expensive', 'cheap', 'value', 'worth', 'money', 'overpriced'],
            'amenities': ['wifi', 'pool', 'parking', 'gym', 'spa', 'facilities', 'ac'],
            'cleanliness': ['clean', 'dirty', 'hygiene', 'tidy', 'spotless', 'maintenance'],
            'noise': ['noise', 'quiet', 'noisy', 'loud', 'peaceful', 'sound']
        }
        
        # Sentiment prefixes
        self.positive_prefix = 'POS_'
        self.negative_prefix = 'NEG_'
        
    def encode_transactions(
        self, 
        texts: List[str],
        df: Optional[pd.DataFrame] = None
    ) -> List[List[str]]:
        """
        Encode texts thành transactions với aspect + sentiment.
        
        Args:
            texts: List of text documents
            df: Optional DataFrame với ratings
            
        Returns:
            List of transactions (lists of items)
        """
        transactions = []
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            items = set()
            
            # Detect aspects
            for aspect, keywords in self.keywords_dict.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        items.add(f'ASPECT_{aspect.upper()}')
                        break
            
            # Detect sentiment words
            positive_words = ['excellent', 'amazing', 'great', 'perfect', 'wonderful', 
                            'fantastic', 'outstanding', 'superb', 'best', 'love', 'recommend']
            negative_words = ['terrible', 'awful', 'horrible', 'bad', 'worst', 'disgusting',
                           'poor', 'disappointed', 'dirty', 'rude', 'avoid', 'not']
            
            text_words = set(text_lower.split())
            has_positive = any(word in text_words for word in positive_words)
            has_negative = any(word in text_words for word in negative_words)
            
            if has_positive:
                items.add('SENTIMENT_POSITIVE')
            if has_negative:
                items.add('SENTIMENT_NEGATIVE')
            
            # Add rating-based sentiment if df provided
            if df is not None and 'rating' in df.columns:
                rating = df.iloc[i]['rating'] if i < len(df) else 3
                if rating >= 4:
                    items.add('RATING_GOOD')
                elif rating <= 2:
                    items.add('RATING_BAD')
            
            transactions.append(list(items))
        
        return transactions


class AssociationMiner:
    """
    Class khai phá luật kết hợp từ hotel reviews.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AssociationMiner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.assoc_config = self.config.get('association_rules', {})
        
        self.min_support = self.assoc_config.get('min_support', 0.01)
        self.min_confidence = self.assoc_config.get('min_confidence', 0.1)
        self.max_itemsets = self.assoc_config.get('max_itemsets', 50)
        self.metric = self.assoc_config.get('metric', 'lift')
        self.min_threshold = self.assoc_config.get('min_threshold', 1.0)
        
        self.transactions = None
        self.frequent_itemsets = None
        self.rules = None
        self.encoder = None
        
    def mine_association_rules(
        self,
        df: pd.DataFrame,
        text_column: str = 'cleaned_text',
        rating_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Khai phá luật kết hợp từ reviews.
        
        Args:
            df: DataFrame chứa dữ liệu
            text_column: Tên cột văn bản đã làm sạch
            rating_column: Tên cột rating (optional)
            
        Returns:
            DataFrame chứa các luật kết hợp
        """
        print("[INFO] Mining association rules...")
        
        # Initialize encoder
        self.encoder = AspectEncoder()
        
        # Get texts
        texts = df[text_column].astype(str).tolist()
        
        # Encode transactions
        self.transactions = self.encoder.encode_transactions(texts, df if rating_column else None)
        
        # Filter empty transactions
        self.transactions = [t for t in self.transactions if len(t) > 0]
        
        print(f"[INFO] Generated {len(self.transactions)} transactions")
        
        if len(self.transactions) < 10:
            print("[WARNING] Not enough transactions for association rules mining")
            return pd.DataFrame()
        
        # Convert to one-hot encoding
        te = TransactionEncoder()
        te_ary = te.fit(self.transactions).transform(self.transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets using Apriori
        print(f"[INFO] Finding frequent itemsets with min_support={self.min_support}...")
        
        try:
            self.frequent_itemsets = apriori(
                df_encoded,
                min_support=self.min_support,
                use_colnames=True,
                max_len=3
            )
        except Exception as e:
            print(f"[WARNING] Apriori error: {e}")
            # Try with lower support
            self.frequent_itemsets = apriori(
                df_encoded,
                min_support=self.min_support / 2,
                use_colnames=True,
                max_len=2
            )
        
        if len(self.frequent_itemsets) == 0:
            print("[WARNING] No frequent itemsets found")
            return pd.DataFrame()
        
        print(f"[INFO] Found {len(self.frequent_itemsets)} frequent itemsets")
        
        # Generate association rules
        self.rules = association_rules(
            self.frequent_itemsets,
            metric=self.metric,
            min_threshold=self.min_threshold
        )
        
        if len(self.rules) == 0:
            print("[WARNING] No association rules found")
            return pd.DataFrame()
        
        # Sort by lift
        self.rules = self.rules.sort_values(by='lift', ascending=False)
        
        print(f"[INFO] Generated {len(self.rules)} association rules")
        
        return self.rules
    
    def get_top_rules(
        self,
        n_rules: int = 20,
        filter_antecedents: Optional[List[str]] = None,
        filter_consequents: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Lấy top rules với optional filtering.
        
        Args:
            n_rules: Số luật cần lấy
            filter_antecedents: Filter theo antecedents
            filter_consequents: Filter theo consequents
            
        Returns:
            DataFrame chứa top rules
        """
        if self.rules is None or len(self.rules) == 0:
            return pd.DataFrame()
        
        rules = self.rules.copy()
        
        # Filter by antecedents
        if filter_antecedents:
            mask = rules['antecedents'].apply(
                lambda x: any(str(item).startswith(f) for f in filter_antecedents for item in x)
            )
            rules = rules[mask]
        
        # Filter by consequents
        if filter_consequents:
            mask = rules['consequents'].apply(
                lambda x: any(str(item).startswith(f) for f in filter_consequents for item in x)
            )
            rules = rules[mask]
        
        # Take top n
        rules = rules.head(n_rules)
        
        # Format for display
        rules_display = rules.copy()
        rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
        
        return rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    
    def analyze_aspect_pairs(self) -> Dict[str, Any]:
        """
        Phân tích các cặp aspect đi kèm nhau.
        
        Returns:
            Dictionary chứa kết quả phân tích
        """
        if self.rules is None:
            return {}
        
        # Find rules about complaints (NEG_) and compliments (POS_)
        analysis = {
            'complaint_rules': [],
            'compliment_rules': [],
            'cross_aspect_rules': []
        }
        
        for _, row in self.rules.iterrows():
            ants = list(row['antecedents'])
            cons = list(row['consequents'])
            
            # Complaint rules: something leads to negative sentiment
            if any('NEG_' in c for c in cons):
                analysis['complaint_rules'].append({
                    'from': ', '.join(ants),
                    'to': ', '.join(cons),
                    'support': row['support'],
                    'confidence': row['confidence'],
                    'lift': row['lift']
                })
            
            # Compliment rules: something leads to positive sentiment
            if any('POS_' in c for c in cons):
                analysis['complaint_rules'].append({
                    'from': ', '.join(ants),
                    'to': ', '.join(cons),
                    'support': row['support'],
                    'confidence': row['confidence'],
                    'lift': row['lift']
                })
            
            # Cross-aspect rules
            has_aspect_ant = any('ASPECT_' in a for a in ants)
            has_aspect_con = any('ASPECT_' in c for c in cons)
            if has_aspect_ant and has_aspect_con:
                analysis['cross_aspect_rules'].append({
                    'from': ', '.join([a for a in ants if 'ASPECT_' in a]),
                    'to': ', '.join([c for c in cons if 'ASPECT_' in c]),
                    'support': row['support'],
                    'confidence': row['confidence'],
                    'lift': row['lift']
                })
        
        return analysis
    
    def interpret_rules(self) -> List[Dict[str, str]]:
        """
        Diễn giải ý nghĩa của các luật.
        
        Returns:
            List of interpretations
        """
        if self.rules is None or len(self.rules) == 0:
            return []
        
        interpretations = []
        
        for _, row in self.rules.head(20).iterrows():
            ant = ', '.join(list(row['antecedents']))
            con = ', '.join(list(row['consequents']))
            
            # Map to readable format
            ant_readable = ant.replace('ASPECT_', '').replace('_', ' ')
            con_readable = con.replace('ASPECT_', '').replace('_', ' ')
            
            interpretation = {
                'rule': f"{ant_readable} -> {con_readable}",
                'support': f"{row['support']:.4f}",
                'confidence': f"{row['confidence']:.4f}",
                'lift': f"{row['lift']:.2f}",
                'meaning': self._interpret_rule_meaning(ant, con, row['lift'])
            }
            
            interpretations.append(interpretation)
        
        return interpretations
    
    def _interpret_rule_meaning(self, antecedent: str, consequent: str, lift: float) -> str:
        """Diễn giải ý nghĩa của một luật."""
        meanings = []
        
        # Parse sentiment
        if 'SENTIMENT_POSITIVE' in consequent:
            meanings.append("When customers mention X, they tend to express positive sentiment")
        elif 'SENTIMENT_NEGATIVE' in consequent:
            meanings.append("When customers mention X, they tend to express negative sentiment")
        
        if 'RATING_GOOD' in consequent:
            meanings.append("Customers mentioning X typically give good ratings (4-5)")
        elif 'RATING_BAD' in consequent:
            meanings.append("Customers mentioning X typically give poor ratings (1-2)")
        
        # Aspect combinations
        aspects_ant = [a for a in antecedent.split(', ') if 'ASPECT_' in a]
        aspects_con = [c for c in consequent.split(', ') if 'ASPECT_' in c]
        
        if aspects_ant and aspects_con:
            meanings.append(f"'{aspects_ant[0]}' is often mentioned with '{aspects_con[0]}'")
        
        if lift > 1.5:
            meanings.append("(Strong association)")
        elif lift > 1:
            meanings.append("(Moderate association)")
        else:
            meanings.append("(Weak association)")
        
        return '. '.join(meanings) if meanings else "General pattern"
    
    def get_support_summary(self) -> Dict[str, float]:
        """
        Lấy tóm tắt support cho các aspects.
        
        Returns:
            Dictionary mapping aspect to support
        """
        if self.frequent_itemsets is None:
            return {}
        
        support_dict = {}
        
        for _, row in self.frequent_itemsets.iterrows():
            items = list(row['itemsets'])
            for item in items:
                if item.startswith('ASPECT_'):
                    support_dict[item] = max(support_dict.get(item, 0), row['support'])
        
        return support_dict
    
    def visualize_top_rules(self, n: int = 10) -> None:
        """Print visualization of top rules."""
        if self.rules is None or len(self.rules) == 0:
            print("No rules to visualize")
            return
        
        print("\n" + "=" * 80)
        print("TOP ASSOCIATION RULES")
        print("=" * 80)
        
        top_rules = self.get_top_rules(n)
        
        for i, row in top_rules.iterrows():
            print(f"\nRule {i+1}:")
            print(f"  IF {row['antecedents']}")
            print(f"  THEN {row['consequents']}")
            print(f"  Support: {row['support']:.4f}")
            print(f"  Confidence: {row['confidence']:.4f}")
            print(f"  Lift: {row['lift']:.2f}")


if __name__ == "__main__":
    print("Testing Association Miner...")
    
    # Sample data
    sample_texts = [
        "The hotel room was clean and comfortable, great service",
        "Terrible service, dirty room, awful breakfast, not recommended",
        "Perfect location, friendly staff, excellent amenities",
        "Great breakfast buffet, clean room, amazing service",
        "Noisy location, poor wifi, small room, but friendly staff",
        "Amazing hotel with wonderful amenities and great location",
        "The bathroom was dirty and the bed was uncomfortable",
        "Excellent location, the staff was very helpful and professional"
    ] * 25
    
    df = pd.DataFrame({
        'cleaned_text': sample_texts,
        'rating': [5, 1, 5, 5, 2, 5, 2, 5] * 25
    })
    
    # Mine association rules
    miner = AssociationMiner()
    rules = miner.mine_association_rules(df)
    
    if len(rules) > 0:
        print(f"\nFound {len(rules)} rules")
        miner.visualize_top_rules(5)
        
        print("\nInterpretations:")
        for interp in miner.interpret_rules()[:5]:
            print(f"  {interp['rule']}")
            print(f"    Meaning: {interp['meaning']}")
