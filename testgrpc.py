import tritonclient.grpc as grpcclient

MODEL_NAME = 'tensorrt_llm_bls'
# Create Triton client
triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

# Check server and model status
print("Is server live:", triton_client.is_server_live())
print("Is server ready:", triton_client.is_server_ready())
print("Is model ready:", triton_client.is_model_ready(MODEL_NAME))

# Get server metadata
server_metadata = triton_client.get_server_metadata()
print("Server Metadata:")
print(server_metadata)

# Get model metadata
model_metadata = triton_client.get_model_metadata(MODEL_NAME)
print("Model Metadata:")
print(model_metadata)

# Get model configuration
model_config = triton_client.get_model_config(MODEL_NAME)
print("Model Config:")
print(model_config)
