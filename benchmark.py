"""Benchmark offline inference throughput using TensorRT-LLM or Triton Inference Server."""

import argparse
import json
import os
import random
import time
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

def sample_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with at least 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def run_trtllm(requests, engine_dir, batch_size, max_output_len):
    import tensorrt_llm
    from tensorrt_llm.bindings.executor import Executor, Request, SamplingConfig
    from tensorrt_llm.bindings.executor import ModelType, ExecutorConfig

    # Load the engine
    engine_path = os.path.join(engine_dir, "engine.trt")
    if not os.path.exists(engine_path):
        print(f"Engine file not found at {engine_path}")
        raise FileNotFoundError(f"Engine file not found at {engine_path}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(engine_dir)

    # Create executor and load model to GPU
    executor = Executor(model_path=engine_path, model_type=ModelType.DECODER_ONLY,
                        executor_config=ExecutorConfig())
    print("Executor initialized.")

    prompts = [prompt for prompt, _, _ in requests]
    start = time.perf_counter()

    total_responses = 0
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        batch_prompts = prompts[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        requests_list = []
        for prompt in batch_prompts:
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids[0].tolist()
            sampling_config = SamplingConfig()
            request = Request(input_token_ids=input_ids, streaming=False, sampling_config=sampling_config,
                              max_new_tokens=max_output_len)
            requests_list.append(request)

        print(f"Processing batch {batch_idx + 1}/{num_batches}")
        executor.enqueue_requests(requests_list)

        # Wait for responses
        responses_received = 0
        total_requests = len(requests_list)
        while responses_received < total_requests:
            responses = executor.get_responses()
            for response in responses:
                if response.has_error():
                    print(f"Error in response {response.request_id}: {response.error_msg}")
                else:
                    output_text = tokenizer.decode(response.output_token_ids, skip_special_tokens=True)
                    print(f"Response {response.request_id}: {output_text}")
                responses_received += 1
                total_responses += 1

    end = time.perf_counter()

    executor.shutdown()

    print("Inference completed.")

    return end - start


def run_triton(requests, server_url, model_name, tokenizer, batch_size, max_output_len):
    import tritonclient.grpc as grpcclient
    import numpy as np

    # Ensure the tokenizer has a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a Triton gRPC client
    triton_client = grpcclient.InferenceServerClient(url=server_url)

    # Prepare requests
    prompts = [prompt for prompt, _, _ in requests]
    total_responses = 0

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    start = time.perf_counter()
    for batch_idx in range(num_batches):
        batch_prompts = prompts[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        # Tokenize inputs
        input_ids_list = []
        max_length = 0
        for prompt in batch_prompts:
            input_ids = tokenizer(prompt, return_tensors='np').input_ids[0]
            input_ids_list.append(input_ids)
            max_length = max(max_length, len(input_ids))

        # Pad inputs
        input_ids_padded = []
        for input_ids in input_ids_list:
            pad_length = max_length - len(input_ids)
            input_ids_padded.append(
                np.pad(input_ids, (0, pad_length), mode='constant', constant_values=tokenizer.pad_token_id)
            )

        # Convert to NumPy array
        input_ids_np = np.array(input_ids_padded, dtype=np.int64)

        # Create Triton inputs
        inputs = [
            grpcclient.InferInput('input_ids', input_ids_np.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids_np)

        # Create Triton outputs
        outputs = [
            grpcclient.InferRequestedOutput('output_ids'),
        ]

        # Set parameters (if needed)
        parameters = {
            'max_output_len': max_output_len,
        }

        # Send request to Triton
        results = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            model_version="",
            request_id="",
            parameters=parameters,
            timeout=None,
        )

        # Process the outputs
        output_data = results.as_numpy('output_ids')
        for i in range(len(batch_prompts)):
            output_ids = output_data[i]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            print(f"Response {total_responses + i}: {output_text}")

        total_responses += len(batch_prompts)
    end = time.perf_counter()

    print("Inference completed.")

    return end - start




def main():
    parser = argparse.ArgumentParser(description="Benchmark the throughput using TensorRT-LLM or Triton Inference Server.")
    parser.add_argument("--backend", type=str, choices=["tensorrt", "triton"], required=True,
                        help="Backend to use: 'tensorrt' or 'triton'.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset.")
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length for each request")
    parser.add_argument("--output-len", type=int, default=None, help="Output length for each request.")
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--max-output-len", type=int, default=256, help="Maximum output length.")
    # Arguments for TensorRT-LLM
    parser.add_argument("--engine-dir", type=str, help="Path to the TensorRT Engine directory.")
    # Arguments for Triton
    parser.add_argument("--server-url", type=str, default='localhost:8000', help="URL of the Triton server.")
    parser.add_argument("--model-name", type=str, help="Name of the model on Triton server.")
    parser.add_argument("--tokenizer-dir", type=str, help="Directory of the tokenizer.")

    args = parser.parse_args()

    random.seed(args.seed)

    # Validate arguments based on the backend
    if args.backend == "tensorrt":
        if not args.engine_dir:
            raise ValueError("engine-dir must be specified for TensorRT backend.")
        tokenizer_dir = args.engine_dir
    elif args.backend == "triton":
        if not args.model_name:
            raise ValueError("model-name must be specified for Triton backend.")
        if not args.tokenizer_dir:
            raise ValueError("tokenizer-dir must be specified for Triton backend.")
        tokenizer_dir = args.tokenizer_dir
    else:
        raise ValueError("Invalid backend specified.")

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        if args.input_len is None or args.output_len is None:
            raise ValueError("input-len and output-len must be specified when dataset is not provided.")
        prompt = "hi" * (args.input_len - 1)
        requests = [
            (prompt, args.input_len, args.output_len) for _ in range(args.num_prompts)
        ]
    else:
        requests = sample_requests(
            args.dataset, args.num_prompts, tokenizer, args.output_len
        )

    if args.backend == "tensorrt":
        elapsed_time = run_trtllm(
            requests, args.engine_dir, args.batch_size, args.max_output_len
        )
    elif args.backend == "triton":
        elapsed_time = run_triton(
            requests, args.server_url, args.model_name, tokenizer, args.batch_size, args.max_output_len
        )
    else:
        raise ValueError("Invalid backend specified.")

    total_num_tokens = sum(
        prompt_len + output_len for _, prompt_len, output_len in requests
    )
    print(
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} tokens/s"
    )


if __name__ == "__main__":
    main()
