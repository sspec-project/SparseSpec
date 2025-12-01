import argparse

from examples.run import main
from examples.run_debug import main as main_debug

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Maximum number of new tokens to generate",
    )

    parser.add_argument(
        "--problem_id",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for the model",
    )

    # specgen
    parser.add_argument(
        "--specgen",
        type=str,
        default="full",
        choices=[
            "full",
            "streaming",
            "topk",
            "topk-history",
            "topk-lambda",
            "topk-lambda-history",
            "quest",
            "oracle",
            "topk-lastq",
            "oracle-shared",
        ],
        help="Enable specgen verification",
    )
    parser.add_argument(
        "--min_budget", type=int, default=512, help="Token budget for the model"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="Temperature for sampling"
    )
    parser.add_argument(
        "--budget_ratio",
        type=float,
        default=0.1,
        help="Token budget ratio for the model",
    )
    parser.add_argument(
        "--verify_stride",
        type=int,
        default=16,
        help="Number of tokens to generate per verify chunk",
    )
    parser.add_argument(
        "--recent_length",
        type=int,
        default=16,
        help="Number of recent tokens to consider",
    )
    parser.add_argument(
        "--enable_sink",
        action="store_true",
        help="Enable the sink tokens",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="aime",
        help="Dataset name",
    )

    args = parser.parse_args()

    from datasets import load_dataset

    if args.dataset == "aime":
        dataset = load_dataset("AI-MO/aimo-validation-aime")
        args.problem = dataset["train"][args.problem_id]["problem"]
        args.output_dir = f"results/aime/{args.problem_id}"
    elif args.dataset == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")
        args.problem = dataset["test"][args.problem_id]["problem"]
        args.output_dir = f"results/math500/{args.problem_id}"
    elif args.dataset == "gpqa-diamond":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
        args.problem = dataset["train"][args.problem_id]["Question"]
        args.output_dir = f"results/gpqa-diamond/{args.problem_id}"
    else:
        raise ValueError("Invalid dataset name")
    if args.debug:
        main_debug(args)
    else:
        main(args)
