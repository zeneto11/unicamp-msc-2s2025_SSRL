"""
MMLU Evaluation Framework for Language Models

A class-based evaluation system for testing language models on MMLU-style 
multiple-choice questions in Portuguese.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------
# Loguru setup
# ---------------------------------------------------------------------
logger.remove()  # Remove default handler
logger.add(
    sink=lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
)
# logger.add("mmlu_eval_gemma-3-1b-it.log", rotation="5 MB",
logger.add("mmlu_eval.log", rotation="5 MB",
           retention="10 days", level="INFO")


# ---------------------------------------------------------------------
# Seed setting function - for reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Model parameters
    model_path: str
    base_model_path: str = None
    lora: bool = False

    # Test data parameters
    test_data_path: str = "data/mmlu_test.csv"
    output_dir: str = "results"

    # Sampling configuration
    sample_size: Optional[int] = None
    random_seed: int = 42

    # Generation parameters
    max_new_tokens_short: int = 1   # Short generation for single token - A, B, C, or D
    max_new_tokens_long: int = 32   # Long generation for full answer explanation
    do_sample_short: bool = False   # Use greedy decoding for short answers
    do_sample_long: bool = True     # Use sampling for long answers
    temperature: float = 0.5        # Temperature for sampling

    # System prompt
    system_prompt: str = (
        "Você é um assistente que responde questões de múltipla escolha em português do Brasil.\n"
        "Responda apenas com UMA opção correta (A, B, C ou D).\n"
    )

    @property
    def model_name(self) -> str:
        """Extract model name from path."""
        return Path(self.model_path).name

    @property
    def output_path(self) -> Path:
        """Generate output path for results."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{self.model_name}_results.csv"


class MMLUEvaluator:
    """
    MMLU-style multiple choice question evaluator for language models.

    Handles model loading, inference, answer extraction, and metric calculation.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize the evaluator with configuration.

        Args:
            config: EvaluationConfig instance with model and test parameters
        """
        self.config = config
        set_seed(self.config.random_seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.results_df = None

    # -----------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------
    def load_model(self) -> None:
        """Load the model and tokenizer onto the device."""
        logger.info(f"Device: {self.device}")
        logger.info(f"Loading model: {self.config.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        ).eval()

        logger.info(f"Model loaded successfully")

    def load_model_lora(self) -> None:
        """Load the base model and apply LoRA adapter."""
        logger.info(f"Device: {self.device}")
        logger.info(f"Loading BASE MODEL + LORA ADAPTER")

        # 1. Load base model
        logger.info(f"Base model: {self.config.base_model_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )

        # 2. Load LoRA adapter
        logger.info(f"Loading LoRA adapter: {self.config.model_path}")

        # Load tokenizer from LoRA model path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path)

        # Apply LoRA adapter
        self.model = PeftModel.from_pretrained(
            self.model,
            self.config.model_path,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            base_model_name_or_path=self.config.base_model_path
        )

        self.model = self.model.eval()
        logger.info(f"Model + LoRA loaded successfully on {self.device}")

    # -----------------------------------------------------------------
    # Prompt formatting
    # -----------------------------------------------------------------
    def format_question(self, row: pd.Series) -> str:
        """
        Format a question row into a prompt.

        Args:
            row: DataFrame row containing question and options

        Returns:
            Formatted prompt string
        """
        subject = row['Subject'].replace('_', ' ').title()
        user_prompt = (
            f"{self.config.system_prompt}"
            f"Assunto: {subject}\n\n"
            f"Pergunta: {row['Question']}\n"
            f"A) {row['A']}\n"
            f"B) {row['B']}\n"
            f"C) {row['C']}\n"
            f"D) {row['D']}\n\n"
            f"Resposta correta:"
        )

        return user_prompt

    # -----------------------------------------------------------------
    # Generation methods
    # -----------------------------------------------------------------
    def generate_response_with_chat_template(self, prompt: str) -> str:
        """
        Generate a response from the model for a given prompt using chat template.

        Args:
            prompt: The input prompt string

        Returns:
            A tuple containing the generated long and short text responses (continuation only)
        """
        messages = [{"role": "user", "content": prompt}]
        model_input = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt").to(self.device)

        # Generation kwargs
        generation_kwargs_long = {
            "max_new_tokens": self.config.max_new_tokens_long,
            "do_sample": self.config.do_sample_long,
            "temperature": self.config.temperature,
        }

        generation_kwargs_short = {
            "max_new_tokens": self.config.max_new_tokens_short,
            "do_sample": self.config.do_sample_short,
        }

        # Generate long and short outputs
        with torch.no_grad():
            output_long = self.model.generate(
                model_input, **generation_kwargs_long)

        with torch.no_grad():
            output_short = self.model.generate(
                model_input, **generation_kwargs_short)

        # Decode only the continuation beyond the prompt
        out_decoded_long = self.tokenizer.decode(
            output_long[0][model_input.shape[1]:],
            skip_special_tokens=True
        ).strip()

        out_decoded_short = self.tokenizer.decode(
            output_short[0][model_input.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return out_decoded_long, out_decoded_short

    def generate_response_seqlogprob(self, prompt: str, row: pd.Series) -> str:
        """
        Generate a response from the model for a given prompt using sequential log-probabilities.

        Args:
            prompt: The input prompt string
            row: DataFrame row containing question and options

        Returns:
            The predicted answer letter (A, B, C, or D)
        """
        messages = [{"role": "user", "content": prompt}]
        prefix_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt").to(self.device)

        options = {}
        for letter in "ABCD":
            target_text = f"{letter}) {row[letter]}"
            target_ids = self.tokenizer(
                target_text,
                add_special_tokens=False,
                return_tensors="pt").to(self.device)["input_ids"]

            with torch.no_grad():
                # feed prefix + target[:-1] to predict target[1:]
                inp = torch.cat([prefix_ids, target_ids[:, :-1]], dim=1)
                out = self.model(inp)
                lp = torch.log_softmax(
                    out.logits[:, -target_ids.shape[1]:, :], dim=-1)
                # gather log-probs of the true next tokens
                tgt = target_ids
                token_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                seq_lp = token_lp.sum().item()
            options[letter] = seq_lp

        pred = max(options, key=options.get)
        return pred

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        """
        Extract the multiple choice answer letter from model output.

        Args:
            text: The model's generated text

        Returns:
            The extracted letter (a-d) in lowercase, or None if not found
        """
        match = re.search(r'\b([a-dA-D])\b', text)
        if match:
            return match.group(1).upper()
        return None

    def evaluate(self) -> pd.DataFrame:
        """
        Run the full evaluation pipeline.

        Returns:
            DataFrame with evaluation results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load test data
        logger.info(f"Loading test data from: {self.config.test_data_path}")
        df_test = pd.read_csv(self.config.test_data_path)

        # Sample if configured
        if self.config.sample_size:
            df_test = df_test.sample(
                n=min(self.config.sample_size, len(df_test)),
                random_state=self.config.random_seed
            )

        logger.info(f"Evaluating on {len(df_test)} questions...")

        results = []
        for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Answering MMLU"):
            # Generate prompt
            prompt = self.format_question(row)

            # Generate responses
            out_chat_long, out_chat_short = self.generate_response_with_chat_template(
                prompt)
            out_seqlogprob = self.generate_response_seqlogprob(prompt, row)

            results.append({
                "subject": row["Subject"],
                "question": row["Question"],
                "A": row["A"],
                "B": row["B"],
                "C": row["C"],
                "D": row["D"],
                "correct": row["Answer"],
                "out_chat_long": out_chat_long,
                "out_extracted_chat_long": self.extract_answer(out_chat_long),
                "out_chat_short": out_chat_short,
                "out_seqlogprob": out_seqlogprob,
            })

        # Process results
        self.results_df = pd.DataFrame(results)

        return self.results_df

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    def save_results(self, output_path: Optional[Path] = None) -> None:
        """
        Save evaluation results to CSV.

        Args:
            output_path: Optional custom output path. Uses config default if None.
        """
        if self.results_df is None:
            raise RuntimeError("No results to save. Run evaluate() first.")

        path = output_path or self.config.output_path
        self.results_df.to_csv(path, index=False)
        logger.info(f"Results saved to: {path}")

    def _compute_accuracy(self, col: str) -> Dict[str, float]:
        """
        Compute accuracy for a given output column.

        Args:
            col: Column name in results_df to evaluate

        Returns:
            Dictionary with accuracy and other metrics
        """
        df = self.results_df
        correct = (df[col].str.upper() == df["correct"].str.upper()).sum()
        total = len(df)
        empty = df[col].isna().sum()
        return {
            "accuracy": 100 * correct / total,
            "correct": correct,
            "total": total,
            "empty": empty,
            "distribution": df[col].value_counts().to_dict(),
        }

    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate evaluation metrics.

        Returns:
            Dictionary with metrics for each evaluation mode
        """
        if self.results_df is None:
            raise RuntimeError("No results to compute metrics on.")
        return {
            "with_chat_long_answer": self._compute_accuracy("out_extracted_chat_long"),
            "with_chat_short_answer": self._compute_accuracy("out_chat_short"),
            "seqlogprob": self._compute_accuracy("out_seqlogprob"),
        }

    def print_metrics(self) -> None:
        """Print evaluation metrics in a formatted way."""
        metrics = self.calculate_metrics()
        for mode, data in metrics.items():
            logger.info(f"\n{'='*50}\nRESULTS ({mode})\n{'='*50}")
            logger.info(f"Accuracy: {data['accuracy']:.2f}% "
                        f"({data['correct']}/{data['total']})")
            logger.info(f"Empty answers: {data['empty']}")
            logger.info(f"Distribution: {data['distribution']}")

    def run(self) -> dict:
        """
        Convenience method to run the complete evaluation pipeline.

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"\n{'='*50}\nEVALUATION\n{'='*50}")
        if self.config.lora:
            self.load_model_lora()
        else:
            self.load_model()
        self.evaluate()
        self.save_results()
        self.print_metrics()

        return self.calculate_metrics()


def main():
    """Main entry point for the evaluation script."""
    # Create configuration
    config = EvaluationConfig(

        # ================================================================
        # gemma-3-1b-it base model
        model_path="models/gemma-3-1b-it",

        # ================================================================",

        # gemma-3-1b-pt + LoRA instruction-tuned on MMLU
        # model_path="models/gemma-3-1b-pt-sft105/best_eval",
        # base_model_path="models/gemma-3-1b-pt",

        # ================================================================

        # gemma-3-1b-pt + Wiki Context + LoRA instruction-tuned on MMLU
        # model_path="models/gemma-3-1b-pt-contextual-e1-ckpt1600-sft2/checkpoint-250",
        # base_model_path="models/gemma-3-1b-pt",

        # ================================================================

        lora=False,
        # sample_size=50,
    )

    # Run evaluation
    evaluator = MMLUEvaluator(config)
    metrics = evaluator.run()

    return metrics


if __name__ == "__main__":
    main()
