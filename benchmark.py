"""Benchmark offline inference throughput using Triton Inference Server."""

import argparse
import json
import os
import random
import sys
import time
from typing import List, Optional

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


def sample_requests(
        dataset_path: str,
        num_requests: int,
) -> List[str]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with at least 1 turn.
    dataset = [data for data in dataset if len(data["conversations"]) >= 1]
    # Only keep the first turn of each conversation.
    dataset = [data["conversations"][0]["value"] for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[str] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        prompt = dataset[i]
        if len(prompt.strip()) == 0:
            # Skip empty prompts
            continue
        filtered_dataset.append(prompt)

    return filtered_dataset


def run_triton(requests, server_url, model_name, batch_size, max_output_len, verbose):
    import tritonclient.grpc as grpcclient

    try:
        triton_client = grpcclient.InferenceServerClient(
            url=server_url, verbose=verbose
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    # Verify server is live and model is ready
    if not triton_client.is_server_live():
        print(f"Failed to connect to Triton server at {server_url}")
        sys.exit(1)

    if not triton_client.is_model_ready(model_name=model_name):
        print(f"Model {model_name} is not ready")
        sys.exit(1)

    # Retrieve model metadata to get input and output names
    try:
        metadata = triton_client.get_model_metadata(model_name=model_name)
    except InferenceServerException as e:
        print(f"Could not retrieve model metadata: {e}")
        sys.exit(1)

    # Get required inputs and outputs
    input_names = [inp.name for inp in metadata.inputs]
    output_names = [out.name for out in metadata.outputs]

    # Required inputs for the ensemble model
    required_inputs = ['text_input', 'max_tokens']

    # Check if required inputs are available
    for req_input in required_inputs:
        if req_input not in input_names:
            print(f"Required input '{req_input}' not found in model inputs.")
            sys.exit(1)

    # Output we will process
    desired_output = 'text_output'
    if desired_output not in output_names:
        print(f"Desired output '{desired_output}' not found in model outputs.")
        sys.exit(1)

    # Prepare requests
    prompts = requests
    total_responses = 0

    num_batches = (len(prompts) + batch_size - 1) // batch_size
    start = time.perf_counter()
    for batch_idx in range(num_batches):
        batch_prompts = prompts[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        # Prepare 'text_input' input
        text_input = np.array(batch_prompts, dtype=object).reshape(-1, 1)

        # Prepare 'max_tokens' input
        max_tokens = np.full((len(batch_prompts), 1), max_output_len, dtype=np.int32)

        # Create the data for optional inputs (e.g., temperature)
        temperature = np.full((len(batch_prompts), 1), 1.0, dtype=np.float32)

        # Initialize inputs
        inputs = []
        input_text = grpcclient.InferInput('text_input', text_input.shape, "BYTES")
        input_text.set_data_from_numpy(text_input)

        input_max_tokens = grpcclient.InferInput('max_tokens', max_tokens.shape, "INT32")
        input_max_tokens.set_data_from_numpy(max_tokens)

        input_temperature = grpcclient.InferInput('temperature', temperature.shape, "FP32")
        input_temperature.set_data_from_numpy(temperature)

        # Append inputs
        inputs.append(input_text)
        inputs.append(input_max_tokens)
        inputs.append(input_temperature)

        # Initialize outputs
        outputs = []
        output_text = grpcclient.InferRequestedOutput('text_output')
        outputs.append(output_text)

        # Send inference request
        try:
            results = triton_client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
            )
        except InferenceServerException as e:
            print(f"Inference failed: {e}")
            continue

        # Process the outputs
        output_data = results.as_numpy('text_output')
        for i in range(len(batch_prompts)):
            output_text = output_data[i][0].decode('utf-8')
            print(f"Response {total_responses + i}: {output_text}")

        total_responses += len(batch_prompts)
    end = time.perf_counter()

    print("Inference completed.")

    return end - start


def main():
    parser = argparse.ArgumentParser(description="Benchmark the throughput using Triton Inference Server.")
    parser.add_argument("--backend", type=str, choices=["tensorrt", "triton"], required=True,
                        help="Backend to use: 'tensorrt' or 'triton'.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset.")
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length for each request")
    parser.add_argument("--output-len", type=int, default=None, help="Output length for each request.")
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--max-output-len", type=int, default=256, help="Maximum output length.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    # Arguments for Triton
    parser.add_argument("--server-url", type=str, default='localhost:8001', help="URL of the Triton server.")
    parser.add_argument("--model-name", type=str, help="Name of the model on Triton server.")

    args = parser.parse_args()

    random.seed(args.seed)

    if args.backend == "triton":
        if not args.model_name:
            raise ValueError("model-name must be specified for Triton backend.")
        tokenizer = None
    else:
        raise ValueError("Invalid backend specified.")

    # Sample the requests.
    if args.dataset is None:
        if args.input_len is None:
            raise ValueError("input-len must be specified when dataset is not provided.")
        prompt = "hi" * (args.input_len)
        requests = [prompt for _ in range(args.num_prompts)]
    else:
        requests = sample_requests(
            args.dataset, args.num_prompts
        )

    if args.backend == "triton":
        elapsed_time = run_triton(
            requests, args.server_url, args.model_name, args.batch_size, args.max_output_len, args.verbose
        )
    else:
        raise ValueError("Invalid backend specified.")

    total_num_chars = sum(len(prompt) for prompt in requests)
    print(
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_chars / elapsed_time:.2f} chars/s"
    )


if __name__ == "__main__":
    main()
