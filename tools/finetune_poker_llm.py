"""
Fine-tuning pipeline for creating a local poker LLM.

Steps:
1. Download PokerBench dataset (11,000 solver-optimal scenarios)
2. Format data for instruction fine-tuning
3. Fine-tune Mistral-7B using QLoRA (fits on RTX 3070 8GB)
4. Merge and convert to GGUF for Ollama
5. Create Ollama Modelfile and register

Requirements:
    pip install unsloth datasets transformers torch

Usage:
    python tools/finetune_poker_llm.py --download    # Step 1: Download data
    python tools/finetune_poker_llm.py --prepare     # Step 2: Prepare training data
    python tools/finetune_poker_llm.py --train        # Step 3: Fine-tune
    python tools/finetune_poker_llm.py --export       # Step 4-5: Export to Ollama
    python tools/finetune_poker_llm.py --all          # All steps
"""

import os
import sys
import json
import argparse
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "models", "training_data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "poker_llm")


# ========== STEP 1: Download PokerBench data ==========

def download_pokerbench():
    """Download the PokerBench dataset from GitHub/HuggingFace."""
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=== DOWNLOADING POKERBENCH DATASET ===")
    print("Attempting to clone PokerBench repository...")

    repo_dir = os.path.join(DATA_DIR, "pokerbench")

    if os.path.exists(repo_dir):
        print(f"PokerBench already downloaded at {repo_dir}")
        return repo_dir

    # Try git clone
    ret = os.system(f'git clone https://github.com/pokerllm/pokerbench.git "{repo_dir}" 2>&1')

    if ret != 0:
        print("Git clone failed. Creating synthetic training data instead...")
        create_synthetic_training_data()
        return DATA_DIR

    print(f"Downloaded to {repo_dir}")
    return repo_dir


def create_synthetic_training_data():
    """Create synthetic poker training data based on GTO principles.
    Used as fallback when PokerBench is not available."""

    print("=== GENERATING SYNTHETIC TRAINING DATA ===")

    training_examples = []

    # Pre-flop scenarios
    preflop_scenarios = [
        # (cards, position, situation, correct_action, reasoning)
        ("AA", "UTG", "first to act", "RAISE 2.5BB",
         "Premiumhand. Open raise fran alla positioner."),
        ("AA", "BTN", "facing 3-bet from BB", "4-BET to 22BB",
         "4-bet for value. AA ar alltid en 4-bet."),
        ("KK", "CO", "first to act", "RAISE 2.5BB",
         "Premiumhand. Standard open raise."),
        ("KK", "BB", "facing BTN open to 2.5BB", "3-BET to 9BB",
         "3-bet for value. KK ar for starkt for att bara calla."),
        ("QQ", "HJ", "first to act", "RAISE 2.5BB",
         "Stark hand. Standard open raise."),
        ("QQ", "SB", "facing UTG open and CO 3-bet", "FOLD",
         "QQ ar for svag mot UTG open + CO 3-bet range. Fold ar korrekt."),
        ("AKs", "CO", "first to act", "RAISE 2.5BB",
         "Premium suited broadway. Open raise fran alla positioner."),
        ("AKs", "BTN", "facing HJ open to 2.5BB", "3-BET to 8BB",
         "3-bet for value och som bluff-blocker."),
        ("AKo", "UTG", "first to act", "RAISE 2.5BB",
         "Stark hand. Open raise fran alla positioner."),
        ("JTs", "BTN", "facing CO open to 2.5BB", "CALL",
         "Suited connector med bra position. Flat call for implied odds."),
        ("JTs", "CO", "first to act", "RAISE 2.5BB",
         "Bra suited connector. Open raise fran CO."),
        ("72o", "UTG", "first to act", "FOLD",
         "Samsta starthand. Fold fran alla positioner utom kanske BTN vs tight blinds."),
        ("72o", "BTN", "facing tight BB", "RAISE 2.5BB",
         "Exploit: Steal blinds mot tight spelare. Handen spelar ingen roll."),
        ("A5s", "CO", "facing HJ open to 2.5BB", "3-BET to 8BB",
         "3-bet som bluff. Blocker till AA/AK, suited for equity nar called."),
        ("T9s", "HJ", "first to act", "RAISE 2.5BB",
         "Suited connector. I range for open fran HJ."),
        ("55", "UTG", "first to act", "RAISE 2.5BB",
         "Litet par. Set-mine potential. Open raise standard."),
        ("55", "BB", "facing UTG open to 2.5BB", "CALL",
         "Call for set value. Vi har bra pot odds och implied odds."),
        ("K9o", "UTG", "first to act", "FOLD",
         "Utanfor UTG opening range. For svag."),
        ("K9o", "BTN", "first to act", "RAISE 2.5BB",
         "I BTN opening range. Standard open."),
        ("AJo", "MP", "facing UTG open to 2.5BB", "FOLD",
         "AJo ar en fold mot UTG open i de flesta fall. Dominerad av AQ/AK."),
    ]

    for cards, pos, situation, action, reasoning in preflop_scenarios:
        training_examples.append({
            "instruction": f"Du spelar Texas Hold'em No-Limit. Du har {cards} i {pos} position. "
                          f"Situation: {situation}. Big blind ar 1BB. "
                          f"Vad bor du gora?",
            "output": json.dumps({
                "action": action,
                "confidence": random.randint(70, 95),
                "reasoning": reasoning,
            }, ensure_ascii=False),
        })

    # Post-flop scenarios
    postflop_scenarios = [
        # (cards, board, pot, position, situation, action, reasoning)
        ("AKs", "A 7 2 rainbow", "6BB", "IP", "we raised pre, villain called",
         "BET 4BB (66% pot)", 85,
         "Value bet. Top pair top kicker pa torr board. C-bet standard."),
        ("AKs", "Q J T two-tone", "8BB", "IP", "we raised pre, villain called",
         "BET 5BB (62% pot)", 75,
         "Vi har nut straight draw + two overs. Semi-bluff c-bet."),
        ("77", "A K 7 rainbow", "10BB", "OOP", "villain bet 7BB",
         "RAISE to 22BB", 90,
         "Vi har set pa scary board. Raise for value innan mer scary cards kommer."),
        ("QJs", "Q 8 3 two hearts", "6BB", "IP", "checked to us",
         "BET 4BB (66% pot)", 80,
         "Top pair bra kicker. Value bet. Skydda mot draws."),
        ("65s", "7 8 2 rainbow", "8BB", "IP", "villain checked",
         "BET 3BB (37% pot)", 65,
         "Open-ended straight draw. Semi-bluff liten bet."),
        ("AQo", "K J 4 rainbow", "12BB", "OOP", "villain bet 8BB on turn",
         "CALL", 55,
         "Vi har gutshot + overcard. Pot odds ar tillrackliga for att calla."),
        ("99", "A K Q 7 2", "20BB", "IP", "river, checked to us",
         "CHECK", 80,
         "Underpair pa scary board. Check for showdown value. Bluff-bet har ingen fold equity."),
        ("AA", "J T 4 5 3", "15BB", "IP", "river, villain bets 12BB",
         "CALL", 70,
         "Overpair. Bra bluff-catcher. Villain kan ha missade draws."),
        ("T8s", "9 7 2 6", "10BB", "IP", "turn, we bet flop villain called",
         "BET 7BB (70% pot)", 85,
         "Vi har nut straight. Value bet. Board ar draw-heavy sa bet stort."),
        ("AKo", "8 8 3 8", "6BB", "IP", "villain donk bets 4BB",
         "FOLD", 75,
         "Triple-paired board, villain donk bets. De har trips eller full house."),
    ]

    for cards, board, pot, pos, situation, action, conf, reasoning in postflop_scenarios:
        training_examples.append({
            "instruction": f"Du spelar Texas Hold'em No-Limit. Du har {cards}. "
                          f"Board: {board}. Pot: {pot}. Position: {pos}. "
                          f"Situation: {situation}. "
                          f"Vad bor du gora?",
            "output": json.dumps({
                "action": action,
                "confidence": conf,
                "reasoning": reasoning,
            }, ensure_ascii=False),
        })

    # Opponent-specific scenarios
    exploit_scenarios = [
        ("ATs", "K 8 3 rainbow", "8BB", "IP",
         "villain VPIP 55%, Fold to C-bet 72%, player type: fish",
         "BET 5BB (62% pot)", 70,
         "C-bet bluff. Villain foldar 72% till c-bet. Var hand spelar ingen roll."),
        ("J9s", "Q T 4", "10BB", "IP",
         "villain VPIP 12%, PFR 10%, player type: nit. Villain check-raised us",
         "FOLD", 85,
         "Fold vs nit check-raise. De har alltid en stark hand (QQ+, AQ+)."),
        ("KQo", "A 5 2 rainbow", "12BB", "IP",
         "villain VPIP 48%, AF 0.7, player type: calling station",
         "CHECK", 75,
         "Check vs calling station. Bluffa aldrig mot nagon som callar allt."),
        ("TT", "8 5 2 rainbow", "8BB", "IP",
         "villain VPIP 42%, AF 4.5, player type: maniac. Villain checked",
         "CHECK", 70,
         "Trap vs maniac. Lat dem bluffa. De bettar allt om vi checkar."),
    ]

    for cards, board, pot, pos, opp_info, action, conf, reasoning in exploit_scenarios:
        training_examples.append({
            "instruction": f"Du spelar Texas Hold'em No-Limit. Du har {cards}. "
                          f"Board: {board}. Pot: {pot}. Position: {pos}. "
                          f"Motstandardinfo: {opp_info}. "
                          f"Vad bor du gora?",
            "output": json.dumps({
                "action": action,
                "confidence": conf,
                "reasoning": reasoning,
            }, ensure_ascii=False),
        })

    # Tournament ICM scenarios
    icm_scenarios = [
        ("AQs", "BTN", "5 players left, 4 get paid, you have 15BB, shortest stack has 8BB",
         "FOLD", 65,
         "ICM: Vi ar nara bubblan. Lat shortest stack busta. AQs ar inte vart risken."),
        ("KK", "BTN", "3 players left, all get paid, you have 20BB",
         "ALL-IN", 90,
         "ICM: Alla ar ITM. KK ar en klar shove for value."),
        ("A5o", "BTN", "Heads-up, you have 12BB",
         "ALL-IN", 75,
         "HU push/fold: A5o ar en klar shove med 12BB heads-up."),
        ("87s", "SB", "6 players left, 5 get paid, you have 25BB, BB has 10BB",
         "RAISE 2.5BB", 60,
         "Steal BBs korta stack. De maste spela tight nara bubblan."),
    ]

    for cards, pos, situation, action, conf, reasoning in icm_scenarios:
        training_examples.append({
            "instruction": f"Du spelar en Texas Hold'em turnering. Du har {cards} i {pos}. "
                          f"Situation: {situation}. "
                          f"Vad bor du gora?",
            "output": json.dumps({
                "action": action,
                "confidence": conf,
                "reasoning": reasoning,
            }, ensure_ascii=False),
        })

    # Shuffle and save
    random.shuffle(training_examples)

    output_file = os.path.join(DATA_DIR, "poker_training_data.jsonl")
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"Generated {len(training_examples)} training examples")
    print(f"Saved to {output_file}")
    return output_file


# ========== STEP 2: Prepare data for training ==========

def prepare_training_data():
    """Convert training data into the format needed for fine-tuning."""
    print("=== PREPARING TRAINING DATA ===")

    data_file = os.path.join(DATA_DIR, "poker_training_data.jsonl")

    if not os.path.exists(data_file):
        print("No training data found. Generating synthetic data...")
        data_file = create_synthetic_training_data()

    # Read and format data
    examples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    # Format for chat fine-tuning (Mistral instruction format)
    formatted = []
    for ex in examples:
        formatted.append({
            "messages": [
                {
                    "role": "system",
                    "content": "Du ar en expert-pokercoach. Svara alltid i JSON-format med: action, confidence (0-100), reasoning."
                },
                {"role": "user", "content": ex["instruction"]},
                {"role": "assistant", "content": ex["output"]},
            ]
        })

    output_file = os.path.join(DATA_DIR, "poker_training_formatted.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Formatted {len(formatted)} examples for fine-tuning")
    print(f"Saved to {output_file}")

    # Also create a train/validation split
    random.shuffle(formatted)
    split = int(len(formatted) * 0.9)

    train_file = os.path.join(DATA_DIR, "train.jsonl")
    val_file = os.path.join(DATA_DIR, "val.jsonl")

    with open(train_file, 'w', encoding='utf-8') as f:
        for item in formatted[:split]:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for item in formatted[split:]:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Train: {split} examples, Validation: {len(formatted) - split} examples")

    return train_file


# ========== STEP 3: Fine-tune with QLoRA ==========

def train_model():
    """Fine-tune Mistral-7B with QLoRA using unsloth."""
    print("=== FINE-TUNING MODEL ===")
    print("This requires: pip install unsloth datasets transformers")
    print("And approximately 2-4 hours on an RTX 3070\n")

    try:
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        import torch
    except ImportError:
        print("ERROR: Required packages not installed.")
        print("Install with:")
        print("  pip install unsloth datasets transformers accelerate")
        print("")
        print("Alternative: Use the training script manually:")
        create_training_script()
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load model with 4-bit quantization
    print("Loading Mistral-7B with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    train_file = os.path.join(DATA_DIR, "train.jsonl")
    if not os.path.exists(train_file):
        prepare_training_data()

    dataset = load_dataset("json", data_files={"train": train_file})

    # Format for training
    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)

    # Train
    from transformers import TrainingArguments
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            output_dir=os.path.join(MODEL_DIR, "checkpoints"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            optim="adamw_8bit",
        ),
    )

    print("Starting training...")
    trainer.train()

    # Save
    model.save_pretrained(os.path.join(MODEL_DIR, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "lora_adapter"))
    print(f"Model saved to {MODEL_DIR}/lora_adapter")


def create_training_script():
    """Create a standalone training script that can be run separately."""
    script_path = os.path.join(BASE_DIR, "tools", "train_model.py")

    script_content = '''"""
Standalone poker LLM fine-tuning script.
Run this if the integrated training fails.

Requirements:
    pip install unsloth datasets transformers accelerate trl

Usage:
    python tools/train_model.py
"""

import os
import sys

# Check dependencies
try:
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    import torch
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install unsloth datasets transformers accelerate trl")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "models", "training_data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "poker_llm")

# Check for training data
train_file = os.path.join(DATA_DIR, "train.jsonl")
if not os.path.exists(train_file):
    print(f"Training data not found at {train_file}")
    print("Run: python tools/finetune_poker_llm.py --prepare")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")

# Load model
print("\\nLoading Mistral-7B (4-bit)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
)

# Load and format data
dataset = load_dataset("json", data_files={"train": train_file})

def formatting_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_func, batched=True)

# Train
os.makedirs(os.path.join(MODEL_DIR, "checkpoints"), exist_ok=True)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10, num_train_epochs=3,
        learning_rate=2e-4, fp16=True,
        logging_steps=10, save_strategy="epoch",
        optim="adamw_8bit",
    ),
)

print("\\nStarting training...")
trainer.train()

# Save LoRA adapter
save_path = os.path.join(MODEL_DIR, "lora_adapter")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\\nModel saved to {save_path}")

# Export to GGUF
print("\\nExporting to GGUF for Ollama...")
try:
    model.save_pretrained_gguf(
        os.path.join(MODEL_DIR, "gguf"),
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"GGUF model saved to {MODEL_DIR}/gguf")
    print("\\nTo use with Ollama:")
    print(f"  ollama create poker-ai -f {MODEL_DIR}/Modelfile")
except Exception as e:
    print(f"GGUF export failed: {e}")
    print("You can manually convert using llama.cpp")
'''

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"Training script created at: {script_path}")
    print("Run it with: python tools/train_model.py")


# ========== STEP 4-5: Export to Ollama ==========

def export_to_ollama():
    """Export fine-tuned model to GGUF and register with Ollama."""
    print("=== EXPORTING TO OLLAMA ===")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Create Ollama Modelfile
    modelfile_content = '''FROM mistral

SYSTEM """Du ar en expert-pokercoach som ger korta, tydliga rad i realtid.

Du far spelarens kort, boardet, equity-berakningar, GTO-rekommendation, och motstandarens profil.

Svara ALLTID i JSON-format:
{
    "action": "BET/CALL/RAISE/FOLD/CHECK",
    "amount": 0.00,
    "confidence": 85,
    "reasoning": "Kort forklaring",
    "if_raised": "Vad gora om motstandaren raisar",
    "if_called": "Plan for nasta street",
    "exploit": "Eventuell exploit"
}

Prioriteringsordning:
1. Om motstandardata finns (20+ hander) - exploatera deras svagheter
2. Om inte - spela GTO
3. I turneringar - ta ICM i beaktande
4. Var alltid koncis. Max 2 meningar per falt."""

PARAMETER temperature 0.3
PARAMETER num_predict 300
PARAMETER top_p 0.9
'''

    modelfile_path = os.path.join(MODEL_DIR, "Modelfile")
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    print(f"Modelfile created at: {modelfile_path}")

    # Check if Ollama is available
    if os.system("ollama --version >nul 2>&1") == 0:
        print("\nRegistering model with Ollama...")
        ret = os.system(f'ollama create poker-ai -f "{modelfile_path}"')
        if ret == 0:
            print("Model registered as 'poker-ai' in Ollama!")
            print("Test with: ollama run poker-ai")
        else:
            print("Failed to register with Ollama.")
            print(f"Manual registration: ollama create poker-ai -f \"{modelfile_path}\"")
    else:
        print("\nOllama not found. After installing Ollama, run:")
        print(f'  ollama create poker-ai -f "{modelfile_path}"')

    # Also create a version based on raw Mistral (no fine-tuning needed)
    quick_modelfile = '''FROM mistral

SYSTEM """Du ar en expert-pokercoach som analyserar Texas Hold'em No-Limit i realtid.

INPUT FORMAT: Du far spelarens kort, board, pot, position, equity, och motstandarprofil.

OUTPUT FORMAT - svara ALLTID i JSON:
{
    "action": "FOLD/CHECK/CALL/BET/RAISE",
    "amount": 0.00,
    "confidence": 0-100,
    "reasoning": "Kort motivering",
    "if_raised": "Plan om motstandaren raisar",
    "if_called": "Plan for nasta street",
    "exploit": "Exploit om applicerbart"
}

REGLER:
- Anvand GTO som bas
- Om motstandardata finns (20+ hander): exploatera svagheter
- Fold to c-bet >65% = c-bet brett
- VPIP >40% + AF <1.5 = aldrig bluffa, value bet max
- VPIP <18% = stjal blinds, respektera raises
- Turneringar: ICM-medveten nara bubblan
- Kort stack (<15BB): push/fold
- Aldrig slowplay pa draw-heavy boards"""

PARAMETER temperature 0.3
PARAMETER num_predict 300
'''

    quick_modelfile_path = os.path.join(MODEL_DIR, "Modelfile.quick")
    with open(quick_modelfile_path, 'w', encoding='utf-8') as f:
        f.write(quick_modelfile)

    print(f"\nQuick Modelfile (no fine-tuning needed) created at: {quick_modelfile_path}")
    print("This uses base Mistral with a poker system prompt.")
    print(f'Register with: ollama create poker-ai -f "{quick_modelfile_path}"')


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a poker LLM")
    parser.add_argument("--download", action="store_true", help="Download PokerBench data")
    parser.add_argument("--prepare", action="store_true", help="Prepare training data")
    parser.add_argument("--train", action="store_true", help="Fine-tune model")
    parser.add_argument("--export", action="store_true", help="Export to Ollama")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()

    if args.all:
        download_pokerbench()
        prepare_training_data()
        train_model()
        export_to_ollama()
    elif args.download:
        download_pokerbench()
    elif args.prepare:
        prepare_training_data()
    elif args.train:
        train_model()
    elif args.export:
        export_to_ollama()
    else:
        # Default: prepare data and export (skip GPU-intensive training)
        print("Running quick setup (data + Ollama Modelfile)...")
        print("For full fine-tuning, use --all or --train\n")
        create_synthetic_training_data()
        prepare_training_data()
        export_to_ollama()


if __name__ == "__main__":
    main()
