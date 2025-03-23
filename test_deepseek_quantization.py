import time
import requests
import psutil
import pynvml
from threading import Thread

# LM Studio API Endpoint
API_URL = "http://127.0.0.1:1236/v1/completions"
MODEL_NAME = "deepseek-r1-distill-qwen-1.5b"

# Define test case
prompt = "Explain how AI will grow in the future."
headers = {"Content-Type": "application/json"}

# Function to get the correct model based on quantization
def get_model_name(quantization):
    if quantization == "4bit":
        return "deepseek-r1-distill-qwen-1.5b:4bit"
    else:
        return "deepseek-r1-distill-qwen-1.5b:8bit"

# Function to measure response time
def measure_response_time(quantization):
    model_name = get_model_name(quantization)
    payload = {"model": model_name, "prompt": prompt, "max_tokens": 100}
    start_time = time.time()
    response = requests.post(API_URL, json=payload, headers=headers)
    elapsed_time = time.time() - start_time
    return elapsed_time, response.json()

# Function to measure throughput (requests per second)
def measure_throughput(quantization, num_requests=10):
    model_name = get_model_name(quantization)
    start_time = time.time()
    for _ in range(num_requests):
        requests.post(API_URL, json={"model": model_name, "prompt": prompt, "max_tokens": 100}, headers=headers)
    elapsed_time = time.time() - start_time
    return num_requests / elapsed_time

# Function to check resource utilization
def get_resource_utilization():
    memory_info = psutil.virtual_memory()
    ram_usage = memory_info.percent
    cpu_usage = psutil.cpu_percent(interval=1)
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    pynvml.nvmlShutdown()
    
    return ram_usage, cpu_usage, gpu_utilization

# Function to measure scalability
def measure_scalability(quantization, num_requests=50, batch_size=5):
    def send_request():
        model_name = get_model_name(quantization)
        requests.post(API_URL, json={"model": model_name, "prompt": prompt, "max_tokens": 100}, headers=headers)
    
    threads = []
    start_time = time.time()
    
    for _ in range(num_requests // batch_size):
        batch_threads = [Thread(target=send_request) for _ in range(batch_size)]
        for thread in batch_threads:
            thread.start()
        for thread in batch_threads:
            thread.join()
    
    return (time.time() - start_time) / num_requests

# Run tests for both 4-bit and 8-bit
results = {}
for quantization in ["4bit", "8bit"]:
    print(f"\n Testing {quantization} model...")
    response_time, output = measure_response_time(quantization)
    throughput = measure_throughput(quantization)
    ram_usage, cpu_usage, gpu_usage = get_resource_utilization()
    scalability_latency = measure_scalability(quantization)
    
    results[quantization] = {
        "Response Time": response_time,
        "Throughput": throughput,
        "RAM Usage": ram_usage,
        "CPU Usage": cpu_usage,
        "GPU Usage": gpu_usage,
        "Scalability (avg latency)": scalability_latency,
        "Sample Response": output
    }

# Print results
print("\nLM Studio Benchmark Results")
for quantization, metrics in results.items():
    print(f"\n--- {quantization.upper()} MODEL ---")
    print(f"Response Time: {metrics['Response Time']:.3f} sec")
    print(f"Throughput: {metrics['Throughput']:.2f} req/sec")
    print(f"RAM Usage: {metrics['RAM Usage']}%")
    print(f"CPU Usage: {metrics['CPU Usage']}%")
    print(f"GPU Usage: {metrics['GPU Usage']}%")
    print(f"Scalability (avg latency): {metrics['Scalability (avg latency)']:.3f} sec/request")
    print(f"\n🔹 Sample Response:", metrics["Sample Response"])

