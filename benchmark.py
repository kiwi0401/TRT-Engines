#!/usr/bin/python

import argparse
import json
import random
import sys
import time
from typing import List
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype
from functools import partial
import queue


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def sample_requests(dataset_path: str, num_requests: int) -> List[str]:
    """Samples a given number of prompts from the dataset."""
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Only keep the first turn of each conversation with at least one turn
    dataset = [data["conversations"][0]["value"] for data in dataset if len(data["conversations"]) >= 1]
    random.shuffle(dataset)

    # Filter out empty prompts and limit to `num_requests`
    return [prompt for prompt in dataset if prompt.strip()][:num_requests]


def prepare_tensor(name, input_array):
    tensor = grpcclient.InferInput(name, input_array.shape, np_to_triton_dtype(input_array.dtype))
    tensor.set_data_from_numpy(input_array)
    return tensor


def run_triton_streaming(triton_client, requests, model_name, batch_size, max_output_len, verbose):
    """Run inference with streaming and inflight batching."""
    user_data = UserData()
    triton_client.start_stream(callback=partial(callback, user_data))

    total_responses = 0
    num_batches = (len(requests) + batch_size - 1) // batch_size
    start_time = time.perf_counter()

    for batch_idx in range(num_batches):
        batch_prompts = requests[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        # Prepare input tensors
        text_input = np.array(batch_prompts, dtype=object).reshape(-1, 1)
        max_tokens = np.full((len(batch_prompts), 1), max_output_len, dtype=np.int32)
        temperature = np.full((len(batch_prompts), 1), 1.0, dtype=np.float32)

        inputs = [
            prepare_tensor("text_input", text_input),
            prepare_tensor("max_tokens", max_tokens),
            prepare_tensor("temperature", temperature),
        ]

        # Send the asynchronous streaming request
        request_id = f"batch-{batch_idx}"
        try:
            triton_client.async_stream_infer(model_name, inputs, request_id=request_id)
        except InferenceServerException as e:
            print(f"Inference request failed for batch {batch_idx}: {e}")
            continue

    responses = []
    while total_responses < len(requests):
        try:
            result = user_data._completed_requests.get(timeout=5)
        except queue.Empty:
            print("Timeout waiting for server response.")
            break

        if isinstance(result, InferenceServerException):
            print("Error received in callback: ", result)
            continue

        output_data = result.as_numpy("text_output")
        for i, output in enumerate(output_data):
            decoded_output = output[0].decode("utf-8")
            responses.append(decoded_output)
            if verbose:
                print(f"Response {total_responses + i}: {decoded_output}")

        total_responses += output_data.shape[0]

    end_time = time.perf_counter()
    triton_client.stop_stream()
    print(f"Inference completed in {end_time - start_time:.2f} seconds.")

    return responses, end_time - start_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark the throughput using Triton Inference Server.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset.")
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length for each request")
    parser.add_argument("--output-len", type=int, default=256, help="Output length for each request.")
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--server-url", type=str, default='localhost:8001', help="URL of the Triton server.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model on Triton server.")

    args = parser.parse_args()
    random.seed(args.seed)

    # Prepare the requests
    if args.dataset:
        requests = sample_requests(args.dataset, args.num_prompts)
    elif args.input_len:
        prompt = "hi" * args.input_len
        requests = [prompt for _ in range(args.num_prompts)]
    else:
        raise ValueError("Either `dataset` or `input_len` must be specified.")

    # Initialize Triton client
    try:
        triton_client = grpcclient.InferenceServerClient(url=args.server_url, verbose=args.verbose)
    except Exception as e:
        print("Triton client creation failed: " + str(e))
        sys.exit(1)

    # Run inference with streaming and inflight batching
    responses, elapsed_time = run_triton_streaming(
        triton_client=triton_client,
        requests=requests,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_output_len=args.output_len,
        verbose=args.verbose
    )

    # Calculate and display throughput
    total_num_chars = sum(len(prompt) for prompt in requests)
    print(
        f"\nThroughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_chars / elapsed_time:.2f} chars/s"
    )


if __name__ == "__main__":
    main()
