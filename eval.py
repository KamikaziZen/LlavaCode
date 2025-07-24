import tree_sitter
from tree_sitter import Language, Parser
import re
from typing import Optional, Tuple, Dict, List, Set
from difflib import SequenceMatcher
from collections import defaultdict
import hashlib
import pandas as pd


class CodeSimilarityComparator:
    """
    A class for comparing code similarity using tree-sitter AST analysis
    combined with exact text matching.
    """
    
    def __init__(self, language: str = "python"):
        """
        Initialize the comparator.
        
        Args:
            language: Programming language ("python", "javascript", "java", etc.)
        """
        self.language = language
        self.parser = None
        self._setup_parser()
        
        # Node types that should be heavily weighted in comparison
        self.structural_nodes = {
            'function_definition', 'class_definition', 'if_statement', 
            'for_statement', 'while_statement', 'try_statement',
            'with_statement', 'assignment', 'call', 'return_statement'
        }
        
        # Node types that should be normalized (less important for comparison)  
        self.normalizable_nodes = {
            'identifier', 'string', 'integer', 'float', 'true', 'false', 'none'
        }
    
    def _setup_parser(self):
        """Setup tree-sitter parser. In real usage, you'd need to install languages."""
        try:
            # This is a placeholder - in real usage you need to:
            # 1. Install tree-sitter languages: pip install tree-sitter-python
            # 2. Build language libraries
            # For demo purposes, we'll simulate this
            self.parser = Parser()
            # self.parser.set_language(Language(tree_sitter.Language.build_library(...)))
        except Exception as e:
            print(f"Warning: Could not setup tree-sitter parser: {e}")
            print("Falling back to text-based comparison only")
            self.parser = None
    
    def _parse_code(self, code: str) -> Optional[any]:
        """Parse code into AST using tree-sitter."""
        if not self.parser:
            return None
            
        try:
            tree = self.parser.parse(bytes(code, "utf8"))
            return tree.root_node
        except Exception:
            # Tree-sitter is very robust, but handle any edge cases
            return None
    
    def _normalize_node(self, node) -> str:
        """
        Normalize a node for comparison by abstracting away non-critical details.
        """
        if not node:
            return ""
            
        node_type = node.type
        
        # For structural nodes, keep the structure but normalize children
        if node_type in self.structural_nodes:
            # Create a canonical representation
            children = []
            for child in node.children:
                children.append(self._normalize_node(child))
            return f"{node_type}({','.join(children)})"
        
        # For normalizable nodes, replace with generic tokens
        elif node_type in self.normalizable_nodes:
            if node_type == 'identifier':
                return 'ID'
            elif node_type in ['string']:
                return 'STR'  
            elif node_type in ['integer', 'float']:
                return 'NUM'
            elif node_type in ['true', 'false']:
                return 'BOOL'
            elif node_type == 'none':
                return 'NULL'
        
        # For other nodes, include type and normalized children
        if hasattr(node, 'children') and node.children:
            children = []
            for child in node.children:
                children.append(self._normalize_node(child))
            return f"{node_type}({','.join(children)})"
        else:
            # Leaf node
            return node_type
    
    def _extract_token_sequence(self, code: str) -> List[str]:
        """Extract a sequence of normalized tokens from code."""
        # Simple tokenization for fallback
        tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
        normalized = []
        
        for token in tokens:
            # Normalize identifiers (but keep keywords)
            if token.isidentifier() and not self._is_keyword(token):
                normalized.append('ID')
            elif token.isdigit() or self._is_float(token):
                normalized.append('NUM')
            elif token.startswith('"') or token.startswith("'"):
                normalized.append('STR')
            else:
                normalized.append(token)
        
        return normalized
    
    def _is_keyword(self, token: str) -> bool:
        """Check if token is a language keyword."""
        python_keywords = {
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 
            'elif', 'else', 'except', 'exec', 'finally', 'for', 'from', 'global',
            'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass', 'print',
            'raise', 'return', 'try', 'while', 'with', 'yield', 'True', 'False', 'None'
        }
        return token in python_keywords
    
    def _is_float(self, token: str) -> bool:
        """Check if token is a float."""
        try:
            float(token)
            return '.' in token
        except ValueError:
            return False
    
    def _tree_similarity(self, code1: str, code2: str) -> float:
        """Calculate AST-based similarity between two code snippets."""
        ast1 = self._parse_code(code1)
        ast2 = self._parse_code(code2)
        
        if not ast1 or not ast2:
            # Fallback to token-based comparison
            return self._token_similarity(code1, code2)
        
        # Normalize both ASTs
        norm1 = self._normalize_node(ast1)
        norm2 = self._normalize_node(ast2)
        
        # Use sequence matching on normalized representations
        matcher = SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()
    
    def _token_similarity(self, code1: str, code2: str) -> float:
        """Calculate token-based similarity as fallback."""
        tokens1 = self._extract_token_sequence(code1)
        tokens2 = self._extract_token_sequence(code2)
        
        matcher = SequenceMatcher(None, tokens1, tokens2)
        return matcher.ratio()
    
    def _exact_similarity(self, code1: str, code2: str) -> float:
        """Calculate exact text similarity."""
        # Normalize whitespace for fairer comparison
        norm1 = re.sub(r'\s+', ' ', code1.strip())
        norm2 = re.sub(r'\s+', ' ', code2.strip())
        
        matcher = SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()
    
    def compare(self, code1: str, code2: str, 
                ast_weight: float = 0.7, 
                exact_weight: float = 0.3) -> Dict[str, float]:
        """
        Compare two code snippets using combined AST and exact similarity.
        
        Args:
            code1: First code snippet
            code2: Second code snippet  
            ast_weight: Weight for AST-based similarity (0-1)
            exact_weight: Weight for exact text similarity (0-1)
            
        Returns:
            Dictionary with similarity scores and details
        """
        # Ensure weights sum to 1
        total_weight = ast_weight + exact_weight
        ast_weight = ast_weight / total_weight
        exact_weight = exact_weight / total_weight
        
        # Calculate individual similarity scores
        ast_sim = self._tree_similarity(code1, code2)
        exact_sim = self._exact_similarity(code1, code2)
        
        # Calculate combined score
        combined_sim = (ast_weight * ast_sim) + (exact_weight * exact_sim)
        
        return {
            'combined_similarity': combined_sim,
            'ast_similarity': ast_sim,
            'exact_similarity': exact_sim,
            'ast_weight': ast_weight,
            'exact_weight': exact_weight
        }
    
    def batch_compare(self, reference: str, candidates: List[str], 
                     **kwargs) -> List[Dict[str, float]]:
        """
        Compare one reference code against multiple candidates.
        
        Args:
            reference: Reference code snippet
            candidates: List of candidate code snippets
            **kwargs: Arguments passed to compare()
            
        Returns:
            List of similarity results, sorted by combined similarity
        """
        results = []
        
        for i, candidate in enumerate(candidates):
            result = self.compare(reference, candidate, **kwargs)
            result['candidate_index'] = i
            results.append(result)
        
        # Sort by combined similarity (descending)
        results.sort(key=lambda x: x['combined_similarity'], reverse=True)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    df = pd.read_json('examples.jsonl', lines=True)

    # Create comparator instance
    comparator = CodeSimilarityComparator()
    
    res = []
    for mem, proj in zip(df['generation w/ mem'].tolist(), df['generation w/ proj'].tolist()):
        res.append(comparator.compare(mem, proj))
    
    for key in res[0].keys():
        print(key, sum((_[key] for _ in res)) / len(res))
