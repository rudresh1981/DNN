"""
CNN Assignment Auto-Grader
Evaluates student submissions for the CNN assignment based on strict requirements.
"""

import json
import re
import sys
import os
from pathlib import Path

class CNNAssignmentEvaluator:
    def __init__(self, notebook_path, folder_name=None):
        self.notebook_path = Path(notebook_path)
        self.folder_name = folder_name
        self.notebook = None
        self.total_marks = 0
        self.breakdown = {}
        self.comments = []
        
    def load_notebook(self):
        """Load and parse the notebook file."""
        try:
            with open(self.notebook_path, 'r', encoding='utf-8') as f:
                self.notebook = json.load(f)
            return True
        except Exception as e:
            self.comments.append(f"Error loading notebook: {str(e)}")
            return False
    
    def pre_validation_check(self):
        """
        PRE_VALIDATION_CHECK_CNN Algorithm
        Returns: (pass/fail, error_message)
        """
        # 1. Check if notebook loaded
        if not self.notebook:
            return False, "Error or corrupted notebook - cannot be opened", 0
        
        # 2. Extract and compare BITS ID
        bits_id_from_filename = self.extract_bits_id_from_filename()
        bits_id_from_notebook = self.extract_bits_id_from_notebook()
        
        if bits_id_from_filename != bits_id_from_notebook:
            return False, f"Filename BITS ID ({bits_id_from_filename}) does not match notebook BITS ID ({bits_id_from_notebook})", 0
        
        # 3. Check folder name vs student name (if provided)
        if self.folder_name:
            student_name = self.extract_student_name()
            if self.folder_name.lower() != student_name.lower():
                return False, f"Folder name ({self.folder_name}) does not match student name ({student_name})", 0
        
        # 4. Check notebook execution status
        if not self.check_execution_status():
            return False, "All outputs cleared - notebook not executed", 0
        
        # 5. Check for execution errors
        if self.has_execution_errors():
            return False, "Notebook contains execution errors", 0
        
        return True, "Pre-validation successful", None
    
    def extract_bits_id_from_filename(self):
        """Extract BITS ID from filename (e.g., 2025AA05787_cnn_assignment.ipynb)."""
        filename = self.notebook_path.stem
        match = re.search(r'(\d{4}[A-Z]{2}\d{5})', filename)
        return match.group(1) if match else None
    
    def extract_bits_id_from_notebook(self):
        """Extract BITS ID from notebook cells."""
        for cell in self.notebook.get('cells', []):
            content = ''.join(cell.get('source', []))
            match = re.search(r'BITS ID:\s*\[?(\d{4}[A-Z]{2}\d{5})\]?', content, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def extract_student_name(self):
        """Extract student name from notebook."""
        for cell in self.notebook.get('cells', []):
            content = ''.join(cell.get('source', []))
            match = re.search(r'Name:\s*\[?([^\]]+)\]?', content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def check_execution_status(self):
        """Check if notebook has been executed (has outputs)."""
        code_cells = [c for c in self.notebook.get('cells', []) if c.get('cell_type') == 'code']
        if not code_cells:
            return False
        
        # Check if at least some cells have outputs
        cells_with_output = sum(1 for c in code_cells if c.get('outputs'))
        return cells_with_output > len(code_cells) * 0.3  # At least 30% of cells should have output
    
    def has_execution_errors(self):
        """Check if any cell contains execution errors."""
        for cell in self.notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'error':
                        return True
                    # Check for error in text output
                    if output.get('output_type') == 'stream':
                        text = ''.join(output.get('text', []))
                        if 'Error' in text or 'Exception' in text or 'Traceback' in text:
                            return True
        return False
    
    def extract_all_source_code(self):
        """Extract all source code from notebook."""
        source_code = []
        for cell in self.notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source_code.append(''.join(cell.get('source', [])))
        return '\n'.join(source_code)
    
    def extract_json_output(self):
        """Extract JSON output from notebook cells."""
        for cell in self.notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'stream':
                        text = ''.join(output.get('text', []))
                        # Try to find JSON content
                        json_match = re.search(r'\{[\s\S]*"dataset_name"[\s\S]*\}', text)
                        if json_match:
                            try:
                                return json.loads(json_match.group(0))
                            except:
                                pass
        return None
    
    def check_custom_cnn(self, json_data):
        """
        CUSTOM_CNN_CHECK (5 marks)
        """
        score = 0
        source_code = self.extract_all_source_code()
        
        if not json_data:
            self.comments.append("No JSON output found")
            return 0
        
        custom_cnn_data = json_data.get('custom_cnn', {})
        
        # a) Check architecture with GAP (2 marks)
        has_conv_layers = 'Conv2D' in source_code or 'nn.Conv2d' in source_code
        has_gap = ('GlobalAveragePooling2D' in source_code or 
                   'AdaptiveAvgPool2d' in source_code or 
                   'global_average_pool' in source_code or
                   custom_cnn_data.get('architecture', {}).get('has_global_average_pooling') == True)
        
        uses_flatten_dense = (('Flatten' in source_code or 'flatten' in source_code) and 
                             ('Dense' in source_code or 'Linear' in source_code) and 
                             not has_gap)
        
        if has_conv_layers and has_gap and not uses_flatten_dense:
            score += 2
        elif uses_flatten_dense:
            self.comments.append("Custom CNN uses Flatten+Dense instead of GAP (prohibited)")
        elif not has_gap:
            self.comments.append("Custom CNN missing Global Average Pooling layer")
        else:
            self.comments.append("Custom CNN architecture incomplete")
        
        # b) Check model compilation/configuration (1 mark)
        framework = custom_cnn_data.get('framework', '').lower()
        properly_configured = False
        
        if framework in ['keras', 'tensorflow']:
            properly_configured = 'model.compile' in source_code or 'compile(' in source_code
        elif framework == 'pytorch':
            properly_configured = (('optimizer' in source_code or 'optim.' in source_code) and 
                                  ('criterion' in source_code or 'loss' in source_code))
        
        if properly_configured:
            score += 1
        else:
            self.comments.append("Custom CNN not properly compiled/configured")
        
        # c) Check training completed (1 mark)
        initial_loss = custom_cnn_data.get('initial_loss')
        final_loss = custom_cnn_data.get('final_loss')
        
        if initial_loss and final_loss and initial_loss > 0 and final_loss > 0:
            score += 1
        else:
            self.comments.append("Custom CNN loss values missing or invalid")
        
        # d) Check metrics calculated (1 mark)
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metrics_found = sum(1 for m in required_metrics 
                          if custom_cnn_data.get(m) is not None and custom_cnn_data.get(m) != 0)
        
        if metrics_found == 4:
            score += 1
        else:
            self.comments.append(f"Custom CNN metrics incomplete: {metrics_found}/4")
        
        return score
    
    def check_transfer_learning(self, json_data):
        """
        TRANSFER_LEARNING_CHECK (5 marks)
        """
        score = 0
        
        if not json_data:
            return 0
        
        tl_data = json_data.get('transfer_learning', {})
        
        # a) Check base model with frozen layers (2 marks)
        base_model = tl_data.get('base_model', '').lower()
        valid_models = ['resnet18', 'resnet50', 'vgg16', 'vgg19']
        
        valid_base_model = any(model in base_model for model in valid_models)
        frozen_layers = tl_data.get('frozen_layers', 0)
        has_frozen_layers = frozen_layers > 0
        
        if valid_base_model and has_frozen_layers:
            score += 2
        elif not valid_base_model:
            self.comments.append(f"Invalid base model: {base_model}")
        elif not has_frozen_layers:
            self.comments.append("Transfer learning base layers not frozen")
        else:
            self.comments.append("Transfer learning base model setup incomplete")
        
        # b) Check GAP + custom head (1 mark)
        has_gap = tl_data.get('has_global_average_pooling') == True
        
        if has_gap:
            score += 1
        else:
            self.comments.append("Transfer learning model missing GAP layer")
        
        # c) Check training completed (1 mark)
        initial_loss = tl_data.get('initial_loss')
        final_loss = tl_data.get('final_loss')
        
        if initial_loss and final_loss and initial_loss > 0 and final_loss > 0:
            score += 1
        else:
            self.comments.append("Transfer learning loss values missing or invalid")
        
        # d) Check metrics calculated (1 mark)
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metrics_found = sum(1 for m in required_metrics 
                          if tl_data.get(m) is not None and tl_data.get(m) != 0)
        
        if metrics_found == 4:
            score += 1
        else:
            self.comments.append(f"Transfer learning metrics incomplete: {metrics_found}/4")
        
        return score
    
    def check_loss_convergence(self, json_data):
        """
        LOSS_CONVERGENCE_CHECK (4 marks)
        """
        score = 0
        
        if not json_data:
            return 0
        
        custom_cnn_data = json_data.get('custom_cnn', {})
        tl_data = json_data.get('transfer_learning', {})
        
        # a) Custom CNN convergence (2 marks)
        cnn_initial = custom_cnn_data.get('initial_loss')
        cnn_final = custom_cnn_data.get('final_loss')
        
        if cnn_initial and cnn_final and cnn_initial > 0:
            if cnn_final < cnn_initial:
                reduction_pct = ((cnn_initial - cnn_final) / cnn_initial) * 100
                
                if reduction_pct >= 50:
                    score += 2
                    self.comments.append(f"Custom CNN converged well: {reduction_pct:.1f}% reduction")
                elif reduction_pct >= 20:
                    score += 1
                    self.comments.append(f"Custom CNN partial convergence: {reduction_pct:.1f}%")
                else:
                    self.comments.append(f"Custom CNN poor convergence: {reduction_pct:.1f}%")
            else:
                self.comments.append("Custom CNN loss did not decrease")
        else:
            self.comments.append("Custom CNN loss values invalid")
        
        # b) Transfer learning convergence (2 marks)
        tl_initial = tl_data.get('initial_loss')
        tl_final = tl_data.get('final_loss')
        
        if tl_initial and tl_final and tl_initial > 0:
            if tl_final < tl_initial:
                reduction_pct = ((tl_initial - tl_final) / tl_initial) * 100
                
                if reduction_pct >= 50:
                    score += 2
                    self.comments.append(f"Transfer learning converged well: {reduction_pct:.1f}% reduction")
                elif reduction_pct >= 20:
                    score += 1
                    self.comments.append(f"Transfer learning partial convergence: {reduction_pct:.1f}%")
                else:
                    self.comments.append(f"Transfer learning poor convergence: {reduction_pct:.1f}%")
            else:
                self.comments.append("Transfer learning loss did not decrease")
        else:
            self.comments.append("Transfer learning loss values invalid")
        
        return score
    
    def check_metrics_validation(self, json_data):
        """
        METRICS_VALIDATION (2 marks)
        """
        score = 0
        
        if not json_data:
            return 0
        
        custom_cnn_data = json_data.get('custom_cnn', {})
        tl_data = json_data.get('transfer_learning', {})
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        cnn_metrics_complete = True
        cnn_metrics_valid = True
        tl_metrics_complete = True
        tl_metrics_valid = True
        
        for metric in required_metrics:
            # Custom CNN
            cnn_val = custom_cnn_data.get(metric)
            if cnn_val is None or cnn_val == 0:
                cnn_metrics_complete = False
            elif not (0 <= cnn_val <= 1):
                cnn_metrics_valid = False
            
            # Transfer Learning
            tl_val = tl_data.get(metric)
            if tl_val is None or tl_val == 0:
                tl_metrics_complete = False
            elif not (0 <= tl_val <= 1):
                tl_metrics_valid = False
        
        # Scoring
        if cnn_metrics_complete and cnn_metrics_valid and tl_metrics_complete and tl_metrics_valid:
            score = 2
        elif (cnn_metrics_complete and cnn_metrics_valid) or (tl_metrics_complete and tl_metrics_valid):
            score = 1
            if not cnn_metrics_complete or not tl_metrics_complete:
                self.comments.append("Metrics incomplete for one or both models")
            else:
                self.comments.append("Metrics out of valid range [0, 1] for one or both models")
        else:
            score = 0
            self.comments.append("Metrics missing or invalid for both models")
        
        # Primary metric validation (informational only)
        primary_metric = json_data.get('primary_metric', '').lower()
        if primary_metric not in ['accuracy', 'precision', 'recall', 'f1-score', 'f1_score']:
            self.comments.append("Warning: Invalid primary metric specified")
        
        metric_justification = json_data.get('metric_justification', '')
        if not metric_justification or len(metric_justification.strip()) < 20:
            self.comments.append("Warning: Metric justification missing or too short")
        
        return score
    
    def check_analysis(self, json_data):
        """
        ANALYSIS_CHECK (2 marks)
        """
        score = 0
        
        if not json_data:
            return 0
        
        analysis_text = json_data.get('analysis', '')
        
        if not analysis_text or len(analysis_text.strip()) < 50:
            # Fallback: try to extract from markdown cells
            for cell in self.notebook.get('cells', []):
                if cell.get('cell_type') == 'markdown':
                    content = ''.join(cell.get('source', []))
                    if 'analysis' in content.lower():
                        analysis_text = content
                        break
        
        word_count = len(analysis_text.split())
        
        if word_count > 200:
            self.comments.append(f"Warning: Analysis exceeds 200 words ({word_count} words)")
        
        # Content quality check
        analysis_lower = analysis_text.lower()
        
        key_topics = [
            'performance', 'accuracy', 'precision', 'recall', 'f1',
            'pre-training', 'pretrained', 'transfer',
            'gap', 'global average pooling', 'overfitting',
            'computational', 'parameters', 'training time', 'cost',
            'convergence', 'loss',
            'advantage', 'disadvantage', 'insight', 'learning'
        ]
        
        topics_covered = sum(1 for topic in key_topics if topic in analysis_lower)
        
        if topics_covered >= 8:
            score = 2
        elif topics_covered >= 5:
            score = 1
            self.comments.append(f"Analysis covers some topics but could be deeper: {topics_covered} keywords")
        else:
            score = 0
            self.comments.append(f"Analysis lacks depth: only {topics_covered} key topics covered")
        
        return score
    
    def check_code_structure(self, json_data):
        """
        CODE_STRUCTURE_CHECK (2 marks)
        """
        score = 0
        source_code = self.extract_all_source_code()
        
        # a) Model definitions complete (1 mark)
        has_cnn = (('Conv2D' in source_code or 'nn.Conv2d' in source_code) and
                   ('GlobalAveragePooling' in source_code or 'AdaptiveAvgPool' in source_code))
        
        has_transfer = (('ResNet' in source_code or 'VGG' in source_code or 
                        'resnet' in source_code or 'vgg' in source_code) and
                       ('trainable' in source_code or 'requires_grad' in source_code or 
                        'freeze' in source_code))
        
        if has_cnn and has_transfer:
            score += 1
        else:
            self.comments.append("Model definitions incomplete")
        
        # b) JSON output structure (1 mark)
        if json_data:
            has_cnn_key = 'custom_cnn' in json_data
            has_tl_key = 'transfer_learning' in json_data
            has_dataset_info = 'dataset_name' in json_data
            
            if has_cnn_key and has_tl_key and has_dataset_info:
                score += 1
            else:
                self.comments.append("JSON output structure incorrect or incomplete")
        else:
            self.comments.append("No JSON output found")
        
        return score
    
    def dataset_checks(self, json_data):
        """
        DATASET_CHECKS (Informational warnings only)
        """
        if not json_data:
            return
        
        # Check train/test split
        train_test_ratio = json_data.get('train_test_ratio', '')
        if train_test_ratio not in ['90/10', '85/15']:
            self.comments.append(f"Warning: Train/test split {train_test_ratio} not standard (use 90/10 or 85/15)")
        
        # Check samples per class
        samples_per_class_info = str(json_data.get('samples_per_class', ''))
        if 'min' in samples_per_class_info.lower():
            numbers = re.findall(r'\d+', samples_per_class_info)
            if numbers and min(int(n) for n in numbers) < 500:
                self.comments.append(f"Warning: Some classes may have < 500 images (minimum required)")
        
        # Check n_classes
        n_classes = json_data.get('n_classes', 0)
        if n_classes < 2 or n_classes > 20:
            self.comments.append(f"Warning: Number of classes ({n_classes}) outside recommended range [2-20]")
    
    def evaluate(self):
        """
        Main evaluation function.
        Returns: (total_marks, breakdown, comments)
        """
        # Load notebook
        if not self.load_notebook():
            return 0, {}, ["Failed to load notebook"]
        
        # Pre-validation check
        passed, message, marks = self.pre_validation_check()
        if not passed:
            return marks if marks is not None else 0, {}, [message]
        
        # Extract JSON output
        json_data = self.extract_json_output()
        
        # Section 1: Custom CNN (5 marks)
        self.breakdown['custom_cnn'] = self.check_custom_cnn(json_data)
        self.total_marks += self.breakdown['custom_cnn']
        
        # Section 2: Transfer Learning (5 marks)
        self.breakdown['transfer_learning'] = self.check_transfer_learning(json_data)
        self.total_marks += self.breakdown['transfer_learning']
        
        # Section 3: Training Process (4 marks)
        self.breakdown['training_process'] = self.check_loss_convergence(json_data)
        self.total_marks += self.breakdown['training_process']
        
        # Section 4: Metrics (2 marks)
        self.breakdown['metrics'] = self.check_metrics_validation(json_data)
        self.total_marks += self.breakdown['metrics']
        
        # Section 5: Analysis (2 marks)
        self.breakdown['analysis'] = self.check_analysis(json_data)
        self.total_marks += self.breakdown['analysis']
        
        # Section 6: Code Structure (2 marks)
        self.breakdown['code_structure'] = self.check_code_structure(json_data)
        self.total_marks += self.breakdown['code_structure']
        
        # Additional dataset checks (informational only)
        self.dataset_checks(json_data)
        
        return self.total_marks, self.breakdown, self.comments


def main():
    """
    Main function to run the evaluator.
    Usage: python evaluate.py <notebook_path> [folder_name]
    """
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <notebook_path> [folder_name]")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    folder_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook file not found: {notebook_path}")
        sys.exit(1)
    
    print("="*70)
    print("CNN ASSIGNMENT AUTO-GRADER")
    print("="*70)
    print(f"Evaluating: {notebook_path}")
    print()
    
    evaluator = CNNAssignmentEvaluator(notebook_path, folder_name)
    total_marks, breakdown, comments = evaluator.evaluate()
    
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nTotal Marks: {total_marks}/20")
    print("\nBreakdown:")
    for section, marks in breakdown.items():
        max_marks = {
            'custom_cnn': 5,
            'transfer_learning': 5,
            'training_process': 4,
            'metrics': 2,
            'analysis': 2,
            'code_structure': 2
        }.get(section, 0)
        print(f"  {section.replace('_', ' ').title()}: {marks}/{max_marks}")
    
    if comments:
        print("\nComments:")
        for i, comment in enumerate(comments, 1):
            print(f"  {i}. {comment}")
    
    print("="*70)
    
    # Save results to JSON
    output_file = notebook_path.replace('.ipynb', '_evaluation.json')
    results = {
        'notebook': notebook_path,
        'total_marks': total_marks,
        'max_marks': 20,
        'breakdown': breakdown,
        'comments': comments
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
