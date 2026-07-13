import argparse
import csv
import json
import os
import random
import re
import zipfile
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


TRAIN_TASKS = [
    "medmcqa",
    "magicoder",
    "finance_alpaca",
    "metamathqa",
    "alpaca_gpt4",
    "e2e_nlg",
]

EVAL_TASKS = [
    "humaneval",
    "gsm8k",
    "phrasebank",
    "arc_c",
    "arc_e",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MALoRA inter-domain datasets as prompt/label DatasetDicts."
    )
    parser.add_argument("--output_root", default="data/malora_interdomain")
    parser.add_argument("--sample_size", type=int, default=30000)
    parser.add_argument("--validation_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--medmcqa_id", default="medmcqa")
    parser.add_argument("--magicoder_id", default="ise-uiuc/Magicoder-OSS-Instruct-75K")
    parser.add_argument("--finance_alpaca_id", default="gbharti/finance-alpaca")
    parser.add_argument("--metamathqa_id", default="meta-math/MetaMathQA")
    parser.add_argument("--alpaca_gpt4_id", default="vicgalle/alpaca-gpt4")
    parser.add_argument("--e2e_nlg_id", default="")
    parser.add_argument(
        "--e2e_nlg_train_file",
        default="https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/trainset.csv",
    )
    parser.add_argument(
        "--e2e_nlg_validation_file",
        default="https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/devset.csv",
    )
    parser.add_argument(
        "--e2e_nlg_test_file",
        default="https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/testset_w_refs.csv",
    )
    parser.add_argument("--humaneval_id", default="openai_humaneval")
    parser.add_argument("--gsm8k_id", default="gsm8k")
    parser.add_argument("--gsm8k_config", default="main")
    parser.add_argument("--phrasebank_id", default="")
    parser.add_argument("--phrasebank_config", default="sentences_75agree")
    parser.add_argument("--phrasebank_file", default="")
    parser.add_argument("--phrasebank_zip", default="")
    parser.add_argument("--arc_id", default="ai2_arc")
    return parser.parse_args()


def load_any(dataset_id, config=None):
    if config:
        return load_dataset(dataset_id, config)
    return load_dataset(dataset_id)


def load_e2e_nlg(args):
    if args.e2e_nlg_id:
        return load_any(args.e2e_nlg_id)
    candidate_sets = [
        {
            "train": args.e2e_nlg_train_file,
            "validation": args.e2e_nlg_validation_file,
            "test": args.e2e_nlg_test_file,
        },
        {
            "train": "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/e2e-dataset/trainset.csv",
            "validation": "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/e2e-dataset/devset.csv",
            "test": "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/e2e-dataset/testset_w_refs.csv",
        },
    ]
    seen = set()
    last_error = None
    for data_files in candidate_sets:
        key = tuple(data_files.items())
        if key in seen:
            continue
        seen.add(key)
        try:
            print(f"[load] e2e_nlg csv train={data_files['train']}")
            return load_dataset("csv", data_files=data_files)
        except Exception as exc:
            last_error = exc
            print(f"[warn] failed to load E2E NLG CSV from {data_files['train']}: {exc}")
    raise RuntimeError(
        "Could not load E2E NLG without dataset scripts. Download trainset.csv, "
        "devset.csv, and testset_w_refs.csv locally, then pass "
        "--e2e_nlg_train_file/--e2e_nlg_validation_file/--e2e_nlg_test_file."
    ) from last_error


def phrasebank_filename(config):
    lowered = (config or "sentences_75agree").lower()
    if "all" in lowered:
        return "Sentences_AllAgree.txt"
    for agreement in ("50", "66", "75"):
        if agreement in lowered:
            return f"Sentences_{agreement}Agree.txt"
    return "Sentences_75Agree.txt"


def decode_text(data):
    for encoding in ("utf-8", "iso-8859-1", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def phrasebank_rows_from_text(text):
    rows = []
    for line in text.splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        if "@" in line:
            sentence, label = line.rsplit("@", 1)
        elif "\t" in line:
            sentence, label = line.rsplit("\t", 1)
        else:
            raise ValueError(
                "Unsupported PhraseBank line format. Expected 'sentence@label'. "
                f"Line starts with: {line[:80]}"
            )
        rows.append({"sentence": sentence.strip(), "label": label.strip().lower()})
    return rows


def phrasebank_rows_from_csv(path):
    rows = []
    for encoding in ("utf-8", "iso-8859-1", "latin-1"):
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lowered = {key.strip().lower(): value for key, value in row.items() if key}
                    sentence = get_value(lowered, ["sentence", "text", "phrase"])
                    label = get_value(lowered, ["label", "sentiment"])
                    if sentence and label:
                        rows.append({"sentence": sentence.strip(), "label": label.strip().lower()})
            return rows
        except UnicodeDecodeError:
            rows = []
            continue
    return rows


def phrasebank_rows_from_file(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        rows = phrasebank_rows_from_csv(path)
        if rows:
            return rows
    return phrasebank_rows_from_text(decode_text(path.read_bytes()))


def phrasebank_rows_from_zip(path, filename):
    target = filename.lower()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        member = next(
            (
                name
                for name in names
                if Path(name).name.lower() == target
            ),
            None,
        )
        if member is None:
            available = ", ".join(Path(name).name for name in names[:20])
            raise FileNotFoundError(f"Could not find {filename} in {path}. First entries: {available}")
        return phrasebank_rows_from_text(decode_text(archive.read(member)))


def load_phrasebank_dataset(args):
    if args.phrasebank_id:
        raw = load_any(args.phrasebank_id, args.phrasebank_config)
        return first_split(raw, ["train", "test", "validation"])

    filename = phrasebank_filename(args.phrasebank_config)
    file_candidates = []
    if args.phrasebank_file:
        phrasebank_file = Path(args.phrasebank_file)
        if phrasebank_file.is_dir():
            file_candidates.append(phrasebank_file / filename)
        else:
            file_candidates.append(phrasebank_file)
    file_candidates.extend(
        [
            Path("/root/GSR-lora/data/raw/financial_phrasebank") / filename,
            Path("/root/GSR-lora/data/raw/financial_phrasebank/FinancialPhraseBank-v1.0") / filename,
            Path("data/raw/financial_phrasebank") / filename,
            Path("data/raw/financial_phrasebank/FinancialPhraseBank-v1.0") / filename,
        ]
    )
    for path in file_candidates:
        if path.exists():
            print(f"[load] phrasebank file={path}")
            return Dataset.from_list(phrasebank_rows_from_file(path))

    zip_candidates = []
    if args.phrasebank_zip:
        zip_candidates.append(Path(args.phrasebank_zip))
    zip_candidates.extend(
        [
            Path("/root/GSR-lora/data/raw/financial_phrasebank/FinancialPhraseBank-v1.0.zip"),
            Path("data/raw/financial_phrasebank/FinancialPhraseBank-v1.0.zip"),
        ]
    )
    for path in zip_candidates:
        if path.exists():
            print(f"[load] phrasebank zip={path}")
            return Dataset.from_list(phrasebank_rows_from_zip(path, filename))

    raise FileNotFoundError(
        "Could not load Financial PhraseBank without dataset scripts. Put "
        f"{filename} or FinancialPhraseBank-v1.0.zip under "
        "/root/GSR-lora/data/raw/financial_phrasebank, or pass "
        "--phrasebank_file/--phrasebank_zip. Do not pass --phrasebank_id unless "
        "your datasets version supports dataset scripts."
    )


def first_split(dataset_dict, names):
    for name in names:
        if name in dataset_dict:
            return dataset_dict[name]
    available = ", ".join(dataset_dict.keys())
    raise KeyError(f"None of splits {names} found. Available splits: {available}")


def sample_dataset(dataset, sample_size, seed):
    if sample_size is None or sample_size <= 0 or len(dataset) <= sample_size:
        return dataset
    return dataset.shuffle(seed=seed).select(range(sample_size))


def make_validation_from_train(dataset, validation_size, seed):
    if validation_size <= 0:
        validation_size = min(1000, len(dataset))
    validation_size = min(validation_size, len(dataset))
    return dataset.shuffle(seed=seed + 17).select(range(validation_size))


def get_value(row, names, default=""):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def normalize_label_text(value):
    return str(value).strip()


def join_instruction(instruction, input_text=""):
    instruction = normalize_label_text(instruction)
    input_text = normalize_label_text(input_text)
    if input_text:
        return f"Instruction: {instruction}\nInput: {input_text}"
    return f"Instruction: {instruction}"


def final_gsm8k_answer(answer):
    text = str(answer)
    if "####" in text:
        return text.split("####")[-1].strip()
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else text.strip()


def choices_to_prompt(question, choices):
    if isinstance(choices, dict):
        labels = choices.get("label") or choices.get("labels") or []
        texts = choices.get("text") or choices.get("texts") or []
    else:
        labels = []
        texts = []
    labels = [str(label) for label in labels]
    texts = [str(text) for text in texts]
    if not labels or len(labels) != len(texts):
        labels = [chr(ord("A") + index) for index in range(len(texts))]
    options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return f"Question: {question}\nChoices:\n{options}\nAnswer with the correct option letter."


def medmcqa_prompt(row):
    options = [
        ("A", get_value(row, ["opa", "option_a"])),
        ("B", get_value(row, ["opb", "option_b"])),
        ("C", get_value(row, ["opc", "option_c"])),
        ("D", get_value(row, ["opd", "option_d"])),
    ]
    option_text = "\n".join(f"{label}. {text}" for label, text in options)
    return f"Medical question: {get_value(row, ['question'])}\nChoices:\n{option_text}\nAnswer with the correct option letter."


def medmcqa_label(row):
    answer = get_value(row, ["cop", "answer", "label"])
    try:
        return ["A", "B", "C", "D"][int(answer)]
    except (ValueError, TypeError, IndexError):
        return str(answer).strip().upper()[:1]


def phrasebank_label(row, dataset):
    value = row.get("label")
    feature = dataset.features.get("label")
    names = getattr(feature, "names", None)
    if names and isinstance(value, int):
        return names[value]
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"negative", "neutral", "positive"}:
            return lowered
    try:
        return {0: "negative", 1: "neutral", 2: "positive"}.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def convert_row(task_name, row, dataset=None):
    if task_name == "medmcqa":
        return {"prompt": medmcqa_prompt(row), "label": medmcqa_label(row)}
    if task_name == "magicoder":
        instruction = get_value(row, ["instruction", "problem", "prompt", "question"])
        input_text = get_value(row, ["input", "starter_code"])
        output = get_value(row, ["response", "output", "solution", "answer"])
        return {"prompt": join_instruction(instruction, input_text), "label": normalize_label_text(output)}
    if task_name == "finance_alpaca":
        instruction = get_value(row, ["instruction", "prompt", "question"])
        input_text = get_value(row, ["input", "context"])
        output = get_value(row, ["output", "response", "answer"])
        return {"prompt": join_instruction(instruction, input_text), "label": normalize_label_text(output)}
    if task_name == "metamathqa":
        prompt = get_value(row, ["query", "question", "problem"])
        answer = get_value(row, ["response", "answer", "solution"])
        return {"prompt": f"Math problem: {prompt}", "label": normalize_label_text(answer)}
    if task_name == "alpaca_gpt4":
        instruction = get_value(row, ["instruction", "prompt"])
        input_text = get_value(row, ["input"])
        output = get_value(row, ["output", "response"])
        return {"prompt": join_instruction(instruction, input_text), "label": normalize_label_text(output)}
    if task_name == "e2e_nlg":
        mr = get_value(row, ["meaning_representation", "mr", "input"])
        reference = get_value(row, ["human_reference", "references", "target", "output", "ref"])
        if isinstance(reference, list):
            reference = reference[0] if reference else ""
        return {"prompt": f"Meaning representation: {mr}\nWrite a fluent description.", "label": normalize_label_text(reference)}
    if task_name == "humaneval":
        prompt = get_value(row, ["prompt"])
        solution = get_value(row, ["canonical_solution"])
        task_id = get_value(row, ["task_id"])
        return {"prompt": prompt, "label": solution, "task_id": task_id}
    if task_name == "gsm8k":
        question = get_value(row, ["question"])
        answer = final_gsm8k_answer(get_value(row, ["answer"]))
        return {"prompt": f"Math word problem: {question}\nAnswer with the final number.", "label": answer}
    if task_name == "phrasebank":
        sentence = get_value(row, ["sentence"])
        label = phrasebank_label(row, dataset)
        return {"prompt": f"Financial sentence: {sentence}\nSentiment options: negative, neutral, positive.", "label": label}
    if task_name in {"arc_c", "arc_e"}:
        question = get_value(row, ["question"])
        prompt = choices_to_prompt(question, row.get("choices", {}))
        label = get_value(row, ["answerKey", "answer", "label"])
        return {"prompt": prompt, "label": str(label).strip()}
    raise ValueError(f"Unsupported task: {task_name}")


def convert_dataset(task_name, dataset):
    rows = []
    for row in tqdm(dataset, desc=f"convert {task_name}"):
        converted = convert_row(task_name, row, dataset=dataset)
        if converted["prompt"] and converted["label"] is not None:
            rows.append(converted)
    return Dataset.from_list(rows)


def save_task(output_root, task_name, splits, overwrite=False):
    output_dir = Path(output_root) / f"{task_name}_with_prompt"
    if output_dir.exists() and not overwrite:
        print(f"[skip] {output_dir}")
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict = DatasetDict(splits)
    dataset_dict.save_to_disk(str(output_dir))
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump({key: len(value) for key, value in splits.items()}, f, indent=2, sort_keys=True)
    print(f"[write] {output_dir} { {key: len(value) for key, value in splits.items()} }")


def task_exists(output_root, task_name):
    return (Path(output_root) / f"{task_name}_with_prompt").exists()


def prepare_train_task(args, task_name, dataset_id, config=None):
    if task_exists(args.output_root, task_name) and not args.overwrite:
        print(f"[skip] {Path(args.output_root) / f'{task_name}_with_prompt'}")
        return
    raw = load_any(dataset_id, config)
    train_raw = first_split(raw, ["train"])
    train_raw = sample_dataset(train_raw, args.sample_size, args.seed)
    try:
        validation_raw = first_split(raw, ["validation", "dev", "test"])
        validation_raw = sample_dataset(validation_raw, args.validation_size, args.seed)
    except KeyError:
        validation_raw = make_validation_from_train(train_raw, args.validation_size, args.seed)
    try:
        test_raw = first_split(raw, ["test", "validation", "dev"])
        test_raw = sample_dataset(test_raw, args.validation_size, args.seed)
    except KeyError:
        test_raw = validation_raw
    splits = {
        "train": convert_dataset(task_name, train_raw),
        "validation": convert_dataset(task_name, validation_raw),
        "test": convert_dataset(task_name, test_raw),
    }
    save_task(args.output_root, task_name, splits, overwrite=args.overwrite)


def prepare_eval_task(args, task_name, dataset_id, config=None, split_names=None):
    if task_exists(args.output_root, task_name) and not args.overwrite:
        print(f"[skip] {Path(args.output_root) / f'{task_name}_with_prompt'}")
        return
    raw = load_any(dataset_id, config)
    split_names = split_names or ["test", "validation", "dev", "train"]
    eval_raw = first_split(raw, split_names)
    splits = {
        "validation": convert_dataset(task_name, eval_raw),
        "test": convert_dataset(task_name, eval_raw),
    }
    if "train" in raw:
        splits["train"] = convert_dataset(task_name, sample_dataset(raw["train"], args.validation_size, args.seed))
    save_task(args.output_root, task_name, splits, overwrite=args.overwrite)


def prepare_e2e_nlg_task(args):
    task_name = "e2e_nlg"
    if task_exists(args.output_root, task_name) and not args.overwrite:
        print(f"[skip] {Path(args.output_root) / f'{task_name}_with_prompt'}")
        return
    raw = load_e2e_nlg(args)
    train_raw = sample_dataset(first_split(raw, ["train"]), args.sample_size, args.seed)
    validation_raw = sample_dataset(first_split(raw, ["validation", "dev", "test"]), args.validation_size, args.seed)
    test_raw = first_split(raw, ["test", "validation", "dev"])
    splits = {
        "train": convert_dataset(task_name, train_raw),
        "validation": convert_dataset(task_name, validation_raw),
        "test": convert_dataset(task_name, test_raw),
    }
    save_task(args.output_root, task_name, splits, overwrite=args.overwrite)


def prepare_phrasebank_task(args):
    task_name = "phrasebank"
    if task_exists(args.output_root, task_name) and not args.overwrite:
        print(f"[skip] {Path(args.output_root) / f'{task_name}_with_prompt'}")
        return
    dataset = load_phrasebank_dataset(args)
    splits = {
        "validation": convert_dataset(task_name, dataset),
        "test": convert_dataset(task_name, dataset),
        "train": convert_dataset(task_name, sample_dataset(dataset, args.validation_size, args.seed)),
    }
    save_task(args.output_root, task_name, splits, overwrite=args.overwrite)


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_root, exist_ok=True)

    prepare_train_task(args, "medmcqa", args.medmcqa_id)
    prepare_train_task(args, "magicoder", args.magicoder_id)
    prepare_train_task(args, "finance_alpaca", args.finance_alpaca_id)
    prepare_train_task(args, "metamathqa", args.metamathqa_id)
    prepare_train_task(args, "alpaca_gpt4", args.alpaca_gpt4_id)
    prepare_e2e_nlg_task(args)

    prepare_eval_task(args, "humaneval", args.humaneval_id, split_names=["test"])
    prepare_eval_task(args, "gsm8k", args.gsm8k_id, args.gsm8k_config, split_names=["test"])
    prepare_phrasebank_task(args)
    prepare_eval_task(args, "arc_c", args.arc_id, "ARC-Challenge", split_names=["test", "validation"])
    prepare_eval_task(args, "arc_e", args.arc_id, "ARC-Easy", split_names=["test", "validation"])

    print(f"[done] MALoRA inter-domain data root: {args.output_root}")
    print(f"[train tasks] {' '.join(TRAIN_TASKS)}")
    print(f"[eval tasks] medmcqa humaneval gsm8k phrasebank arc_c arc_e e2e_nlg")


if __name__ == "__main__":
    main()
