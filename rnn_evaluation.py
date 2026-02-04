"""
================================================================================
RNN ASSIGNMENT EVALUATOR
================================================================================
Deep Neural Networks - Assignment 3: RNN vs Transformer for Time Series

This script evaluates student submissions for the RNN assignment.

Usage:
    python rnn_evaluation.py <notebook_file.ipynb>

Example:
    python rnn_evaluation.py 2025AA05787_rnn_assignment.ipynb

Grading Breakdown (Total: 12 marks):
    - Part 2: LSTM/GRU Implementation (5 marks)
    - Part 3: Transformer Implementation (5 marks)
    - Part 5: Analysis (2 marks)
    - Part 1, 4, 6: Informational (0 marks)

================================================================================
"""

import json
import sys
import os
import re
from pathlib import Path


class RNNAssignmentEvaluator:
    """Evaluates RNN assignment submissions"""
    
    def __init__(self, notebook_path):
        self.notebook_path = notebook_path
        self.notebook_name = os.path.basename(notebook_path)
        self.notebook_data = None
        self.errors = []
        self.warnings = []
        self.marks = {
            'rnn_implementation': 0,
            'transformer_implementation': 0,
            'analysis': 0,
            'total': 0
        }
        self.max_marks = {
            'rnn_implementation': 5,
            'transformer_implementation': 5,
            'analysis': 2,
            'total': 12
        }
        
    def load_notebook(self):
        """Load and parse the Jupyter notebook"""
        try:
            with open(self.notebook_path, 'r', encoding='utf-8') as f:
                self.notebook_data = json.load(f)
            return True
        except FileNotFoundError:
            self.errors.append(f"❌ File not found: {self.notebook_path}")
            return False
        except json.JSONDecodeError:
            self.errors.append(f"❌ Invalid notebook file (corrupted JSON)")
            return False
        except Exception as e:
            self.errors.append(f"❌ Error loading notebook: {str(e)}")
            return False
    
    def get_all_code(self):
        """Extract all code cells from notebook"""
        code_cells = []
        try:
            for cell in self.notebook_data.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        code_cells.append(''.join(source))
                    else:
                        code_cells.append(source)
        except Exception as e:
            self.errors.append(f"⚠️ Error extracting code: {str(e)}")
        return '\n\n'.join(code_cells)
    
    def check_filename_format(self):
        """Check if filename matches required format"""
        # Normalize filename (remove extra whitespace)
        normalized_name = ' '.join(self.notebook_name.split())
        
        # Pattern: starts with 10-13 digits, then _rnn_assignment.ipynb
        pattern = r'^\d{10,13}_rnn_assignment\.ipynb$'
        if not re.match(pattern, normalized_name):
            self.warnings.append(
                f"⚠️ FILENAME WARNING: File should be named '<BITS_ID>_rnn_assignment.ipynb'\n"
                f"   Current: {self.notebook_name}\n"
                f"   Recommended: 2025AA05787_rnn_assignment.ipynb"
            )
            # Don't fail on this - just warn
        return True
    
    def check_execution(self):
        """Check if all cells have been executed"""
        unexecuted_cells = 0
        total_code_cells = 0
        
        try:
            for cell in self.notebook_data.get('cells', []):
                if cell.get('cell_type') == 'code':
                    total_code_cells += 1
                    execution_count = cell.get('execution_count')
                    outputs = cell.get('outputs', [])
                    
                    # Check if cell was never executed
                    if execution_count is None and len(outputs) == 0:
                        unexecuted_cells += 1
        except Exception as e:
            self.warnings.append(f"⚠️ Could not verify execution: {str(e)}")
            return
        
        if unexecuted_cells > 0:
            self.errors.append(
                f"❌ EXECUTION ERROR: {unexecuted_cells}/{total_code_cells} cells not executed\n"
                f"   Run: Kernel → Restart & Run All before submission"
            )
        else:
            print(f"✓ All {total_code_cells} code cells executed")
    
    def check_student_info(self):
        """Verify student information is filled"""
        code = self.get_all_code()
        
        # Extract BITS ID from filename
        filename_bits_id = re.match(r'^(\d{10,13})', self.notebook_name)
        if filename_bits_id:
            filename_bits_id = filename_bits_id.group(1)
        
        # Check if student info is still placeholder
        if '[2025AA05A87]' in code or '[RUDRESH R]' in code:
            self.warnings.append(
                "⚠️ Student information appears to be placeholder/template\n"
                "   Update BITS ID, Name, and Email in cell 2"
            )
        
        # Extract BITS ID from notebook
        bits_id_match = re.search(r'BITS ID:?\s*\[?(\d{10,13})\]?', code)
        if bits_id_match:
            notebook_bits_id = bits_id_match.group(1)
            if filename_bits_id and notebook_bits_id != filename_bits_id:
                self.errors.append(
                    f"❌ BITS ID MISMATCH:\n"
                    f"   Filename: {filename_bits_id}\n"
                    f"   Notebook: {notebook_bits_id}"
                )
    
    def evaluate_rnn_implementation(self):
        """Evaluate LSTM/GRU implementation (5 marks)"""
        code = self.get_all_code()
        marks = 0
        details = []
        
        # Check 1: RNN model function (LSTM or GRU) - 2 marks
        has_rnn_function = False
        has_stacked_layers = False
        
        if 'def build_rnn_model' in code:
            has_rnn_function = True
            marks += 1
            details.append("✓ RNN model building function defined (1 mark)")
            
            # Check for stacked layers
            if ('return_sequences=True' in code and 
                ('LSTM' in code or 'GRU' in code)):
                has_stacked_layers = True
                marks += 1
                details.append("✓ Stacked RNN layers implemented (1 mark)")
            else:
                details.append("✗ Missing stacked layers with return_sequences=True")
        else:
            details.append("✗ Missing build_rnn_model() function")
        
        # Check 2: Model compilation - 1 mark
        if 'model.compile' in code or 'compile(' in code:
            if ('optimizer' in code and 'loss' in code):
                marks += 1
                details.append("✓ Model properly compiled with optimizer and loss (1 mark)")
            else:
                details.append("✗ Compilation incomplete (missing optimizer or loss)")
        else:
            details.append("✗ Model not compiled")
        
        # Check 3: Training with loss tracking - 1 mark
        has_training = False
        has_loss_tracking = False
        
        if 'model.fit' in code or 'fit(' in code:
            has_training = True
            
            if ('rnn_initial_loss' in code and 'rnn_final_loss' in code):
                has_loss_tracking = True
                marks += 1
                details.append("✓ Training completed with loss tracking (1 mark)")
            else:
                details.append("✗ Missing loss tracking (rnn_initial_loss, rnn_final_loss)")
        else:
            details.append("✗ Model not trained (missing model.fit)")
        
        # Check 4: Metrics calculation - 1 mark
        metrics_found = 0
        required_metrics = ['rnn_mae', 'rnn_rmse', 'rnn_mape', 'rnn_r2']
        
        for metric in required_metrics:
            if metric in code:
                metrics_found += 1
        
        if metrics_found == 4:
            marks += 1
            details.append("✓ All 4 metrics calculated (MAE, RMSE, MAPE, R²) (1 mark)")
        elif metrics_found > 0:
            details.append(f"⚠️ Only {metrics_found}/4 metrics calculated")
        else:
            details.append("✗ Metrics not calculated")
        
        self.marks['rnn_implementation'] = marks
        return marks, details
    
    def evaluate_transformer_implementation(self):
        """Evaluate Transformer implementation (5 marks)"""
        code = self.get_all_code()
        marks = 0
        details = []
        
        # Check 1: Positional encoding implementation - 1 mark (CRITICAL)
        has_pos_encoding = False
        pos_encoding_quality = 0
        
        if 'def positional_encoding' in code:
            has_pos_encoding = True
            
            # Check for sinusoidal implementation
            if ('sin' in code.lower() and 'cos' in code.lower() and 
                '10000' in code):
                marks += 1
                pos_encoding_quality = 2
                details.append("✓ Positional encoding implemented (sinusoidal) (1 mark)")
            else:
                pos_encoding_quality = 1
                details.append("⚠️ Positional encoding function exists but may be incomplete")
        else:
            details.append("✗ CRITICAL: Missing positional encoding function")
        
        # Check 2: Positional encoding ADDED to model - CRITICAL
        pos_encoding_added = False
        if has_pos_encoding:
            # Check if positional encoding is actually added to the model
            if (('+ positional_encoding' in code or 
                 'x + pos_enc' in code or 
                 'x = x + pos' in code) and 
                'MultiHeadAttention' in code):
                pos_encoding_added = True
                details.append("✓ Positional encoding ADDED to model architecture")
            else:
                self.warnings.append(
                    "⚠️ WARNING: Positional encoding function exists but may not be added to model\n"
                    "   Ensure: x = x + positional_encoding(...) BEFORE MultiHeadAttention"
                )
        
        # Check 3: Transformer architecture - 2 marks
        has_attention = False
        has_feedforward = False
        
        if 'MultiHeadAttention' in code:
            has_attention = True
            marks += 1
            details.append("✓ Multi-head attention layer used (1 mark)")
        else:
            details.append("✗ Missing MultiHeadAttention layer")
        
        # Check for feed-forward network and residual connections
        if (('Dense(' in code and 'activation=' in code) or 
            ('Linear(' in code and 'relu' in code.lower())):
            if 'LayerNormalization' in code or 'LayerNorm' in code:
                has_feedforward = True
                marks += 1
                details.append("✓ Feed-forward network with layer norm (1 mark)")
            else:
                details.append("⚠️ Feed-forward network found but missing layer normalization")
        else:
            details.append("✗ Missing feed-forward network")
        
        # Check 4: Training with loss tracking - 1 mark
        if ('transformer_model.fit' in code or 
            'history_transformer' in code):
            if ('transformer_initial_loss' in code and 
                'transformer_final_loss' in code):
                marks += 1
                details.append("✓ Training completed with loss tracking (1 mark)")
            else:
                details.append("✗ Missing loss tracking (transformer_initial_loss, transformer_final_loss)")
        else:
            details.append("✗ Transformer not trained")
        
        # Check 5: Metrics calculation - 1 mark
        metrics_found = 0
        required_metrics = ['transformer_mae', 'transformer_rmse', 
                          'transformer_mape', 'transformer_r2']
        
        for metric in required_metrics:
            if metric in code:
                metrics_found += 1
        
        if metrics_found == 4:
            marks += 1
            details.append("✓ All 4 metrics calculated (1 mark)")
        elif metrics_found > 0:
            details.append(f"⚠️ Only {metrics_found}/4 metrics calculated")
        else:
            details.append("✗ Metrics not calculated")
        
        # CRITICAL CHECK: If positional encoding not properly implemented, cap marks
        if not has_pos_encoding or pos_encoding_quality < 2:
            if marks > 0:
                self.warnings.append(
                    "⚠️ CRITICAL: Positional encoding implementation incomplete\n"
                    "   This is MANDATORY for Transformer marks"
                )
        
        if not pos_encoding_added and has_pos_encoding:
            if marks > 1:
                self.warnings.append(
                    "⚠️ CRITICAL: Positional encoding may not be added to model\n"
                    "   Verify: x = x + positional_encoding(...) in model"
                )
        
        self.marks['transformer_implementation'] = marks
        return marks, details
    
    def evaluate_analysis(self):
        """Evaluate analysis quality (2 marks)"""
        code = self.get_all_code()
        marks = 0
        details = []
        
        # Find analysis text (supports both regular strings and f-strings)
        # Try f-string format first: analysis_text = f"""..."""
        analysis_match = re.search(
            r'analysis_text\s*=\s*f?[\'\"]{3}(.*?)[\'\"]{3}',
            code,
            re.DOTALL
        )
        
        if not analysis_match:
            details.append("✗ No analysis_text variable found")
            self.marks['analysis'] = 0
            return 0, details
        
        analysis_text = analysis_match.group(1).strip()
        
        # Check if still TODO
        if 'TODO' in analysis_text:
            details.append("✗ Analysis still contains TODO placeholders")
            self.marks['analysis'] = 0
            return 0, details
        
        # Count words
        words = analysis_text.split()
        word_count = len([w for w in words if w.strip()])
        
        if word_count < 50:
            details.append(f"✗ Analysis too short ({word_count} words, minimum ~150)")
            self.marks['analysis'] = 0
            return 0, details
        
        # Check for key topics (quality-based grading)
        key_topics = {
            'performance': ['metric', 'mape', 'mae', 'rmse', 'r2', 'accuracy', 'error'],
            'architecture': ['architecture', 'layer', 'structure', 'model'],
            'attention': ['attention', 'self-attention', 'multi-head'],
            'dependencies': ['dependency', 'dependencies', 'temporal', 'sequence'],
            'computation': ['time', 'training', 'parameter', 'computational', 'cost'],
            'convergence': ['loss', 'convergence', 'training', 'epoch']
        }
        
        topics_covered = 0
        for topic, keywords in key_topics.items():
            if any(keyword in analysis_text.lower() for keyword in keywords):
                topics_covered += 1
        
        # Grading based on depth
        if topics_covered >= 5:
            marks = 2
            details.append(f"✓ Excellent analysis covering {topics_covered}/6 key topics (2 marks)")
            details.append(f"  Word count: {word_count} words")
        elif topics_covered >= 3:
            marks = 1
            details.append(f"✓ Good analysis covering {topics_covered}/6 key topics (1 mark)")
            details.append(f"  Word count: {word_count} words")
        else:
            marks = 0
            details.append(f"✗ Superficial analysis (only {topics_covered}/6 topics)")
            details.append(f"  Word count: {word_count} words")
        
        if word_count > 250:
            details.append(f"  ⚠️ Note: Analysis exceeds 200 word guideline ({word_count} words)")
        
        self.marks['analysis'] = marks
        return marks, details
    
    def check_dataset_requirements(self):
        """Check dataset requirements"""
        code = self.get_all_code()
        
        # Extract n_samples
        n_samples_match = re.search(r'n_samples\s*=\s*(\d+)', code)
        if n_samples_match:
            n_samples = int(n_samples_match.group(1))
            if n_samples < 1000:
                self.warnings.append(
                    f"⚠️ Dataset has only {n_samples} samples (minimum 1000 required)"
                )
            else:
                print(f"✓ Dataset size: {n_samples} samples (≥1000 required)")
        
        # Extract sequence_length
        seq_length_match = re.search(r'sequence_length\s*=\s*(\d+)', code)
        if seq_length_match:
            seq_length = int(seq_length_match.group(1))
            if seq_length < 10 or seq_length > 50:
                self.warnings.append(
                    f"⚠️ Sequence length {seq_length} outside recommended range (10-50)"
                )
            else:
                print(f"✓ Sequence length: {seq_length} (10-50 range)")
        
        # Check for temporal split
        if 'shuffle=false' in code.lower() or 'temporal' in code.lower():
            print("✓ Temporal train/test split (no shuffling)")
        else:
            self.warnings.append(
                "⚠️ Cannot verify temporal split (no shuffling)\n"
                "   Ensure train/test split preserves time order"
            )
    
    def check_framework(self):
        """Identify framework used"""
        code = self.get_all_code()
        
        if 'tensorflow' in code.lower() or 'keras' in code.lower():
            print("✓ Framework: TensorFlow/Keras")
            return 'keras'
        elif 'torch' in code.lower() or 'pytorch' in code.lower():
            print("✓ Framework: PyTorch")
            return 'pytorch'
        else:
            self.warnings.append("⚠️ Cannot identify framework (Keras or PyTorch)")
            return 'unknown'
    
    def print_report(self):
        """Print comprehensive evaluation report"""
        print("\n" + "="*80)
        print("RNN ASSIGNMENT EVALUATION REPORT")
        print("="*80)
        print(f"Notebook: {self.notebook_name}")
        print(f"Date: {Path(self.notebook_path).stat().st_mtime}")
        print("="*80)
        
        # Critical Errors
        if self.errors:
            print("\n🚨 CRITICAL ERRORS (Auto-fail):")
            print("-" * 80)
            for error in self.errors:
                print(error)
            print("\n⚠️ SUBMISSION WILL RECEIVE 0 MARKS DUE TO CRITICAL ERRORS")
            print("="*80)
            return
        
        # Warnings
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            print("-" * 80)
            for warning in self.warnings:
                print(warning)
        
        # Marks breakdown
        print("\n📊 MARKS BREAKDOWN:")
        print("="*80)
        
        # Part 2: RNN Implementation
        rnn_marks, rnn_details = self.evaluate_rnn_implementation()
        print(f"\nPart 2: LSTM/GRU Implementation")
        print(f"Marks: {rnn_marks}/{self.max_marks['rnn_implementation']}")
        print("-" * 80)
        for detail in rnn_details:
            print(f"  {detail}")
        
        # Part 3: Transformer Implementation
        transformer_marks, transformer_details = self.evaluate_transformer_implementation()
        print(f"\nPart 3: Transformer Implementation")
        print(f"Marks: {transformer_marks}/{self.max_marks['transformer_implementation']}")
        print("-" * 80)
        for detail in transformer_details:
            print(f"  {detail}")
        
        # Part 5: Analysis
        analysis_marks, analysis_details = self.evaluate_analysis()
        print(f"\nPart 5: Analysis")
        print(f"Marks: {analysis_marks}/{self.max_marks['analysis']}")
        print("-" * 80)
        for detail in analysis_details:
            print(f"  {detail}")
        
        # Total
        total_marks = (self.marks['rnn_implementation'] + 
                      self.marks['transformer_implementation'] + 
                      self.marks['analysis'])
        self.marks['total'] = total_marks
        
        print("\n" + "="*80)
        print(f"TOTAL MARKS: {total_marks}/{self.max_marks['total']}")
        print("="*80)
        
        # Grade
        percentage = (total_marks / self.max_marks['total']) * 100
        if percentage >= 90:
            grade = "A+ (Excellent)"
        elif percentage >= 80:
            grade = "A (Very Good)"
        elif percentage >= 70:
            grade = "B+ (Good)"
        elif percentage >= 60:
            grade = "B (Satisfactory)"
        elif percentage >= 50:
            grade = "C (Pass)"
        else:
            grade = "F (Fail)"
        
        print(f"\nPercentage: {percentage:.1f}%")
        print(f"Grade: {grade}")
        
        # Recommendations
        if total_marks < self.max_marks['total']:
            print("\n💡 RECOMMENDATIONS FOR IMPROVEMENT:")
            print("-" * 80)
            
            if rnn_marks < self.max_marks['rnn_implementation']:
                print("• Complete RNN implementation with stacked layers and full metrics")
            
            if transformer_marks < self.max_marks['transformer_implementation']:
                print("• Ensure positional encoding is properly implemented AND added to model")
                print("• Use MultiHeadAttention and implement feed-forward networks")
            
            if analysis_marks < self.max_marks['analysis']:
                print("• Write comprehensive analysis covering 5+ key topics")
                print("• Include specific metrics, architecture comparison, and insights")
        
        print("\n" + "="*80)
        print("✓ Evaluation Complete")
        print("="*80)
    
    def evaluate(self):
        """Run complete evaluation"""
        print("="*80)
        print("RNN ASSIGNMENT EVALUATOR")
        print("="*80)
        print(f"\nEvaluating: {self.notebook_name}")
        print("-" * 80)
        
        # Load notebook
        if not self.load_notebook():
            self.print_report()
            return False
        
        print("✓ Notebook loaded successfully")
        
        # Critical checks
        print("\n🔍 CRITICAL CHECKS:")
        print("-" * 80)
        
        if not self.check_filename_format():
            self.print_report()
            return False
        print(f"✓ Filename format correct")
        
        self.check_student_info()
        self.check_execution()
        
        # If critical errors found, stop
        if self.errors:
            self.print_report()
            return False
        
        # Additional checks
        print("\n🔍 ADDITIONAL CHECKS:")
        print("-" * 80)
        self.check_framework()
        self.check_dataset_requirements()
        
        # Print full report
        self.print_report()
        
        return True


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python rnn_evaluation.py <notebook_file.ipynb>")
        print("\nExample:")
        print("  python rnn_evaluation.py 2025AA05787_rnn_assignment.ipynb")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    
    if not os.path.exists(notebook_path):
        print(f"❌ Error: File not found: {notebook_path}")
        sys.exit(1)
    
    evaluator = RNNAssignmentEvaluator(notebook_path)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
