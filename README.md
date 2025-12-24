# LlavaCode

## Environment Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment

```bash
source .venv/bin/activate
```

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Data Preparation

Unzip the datasets archives into your working directory.

```stack_java_test_1k``` - subset of 1k samples randomly chosen from the test split of TheStackv2 Java dataset.
```python_line_completion_sparse_bm25.jsonl``` - repoeval benchmark (as described in RepoCoder paper)
```java_line_completion_sparse_bm25.jsonl``` - repoeval benchmark for Java

## Checkpoint Preparation

Unzip the provided checkpoints archive and place the `.ckpt` or `.pth` files in the ./ckpt directory.

The expected directory structure is:

```
./stack2_java_test_1k
./python_line_completion_sparse_bm25.jsonl
./java_line_completion_sparse_bm25.jsonl
./ckpt/qwen7_unixcoder_3l_java_2emes_stack2.ckpt
./ckpt/qwen7_unixcoder_3l_java_2emes_stack2_projector.pth
./ckpt/qwen7_unixcoder_3l_python_emes_stack.ckpt
./ckpt/qwen7_unixcoder_3l_python_emes_stack_projector.pth
./run_java_benchmark.sh
./run_java_stack2.sh
./run_python_benchmark.sh
./LlavaCode_inference.py
...
```


## Running a single example

To run a single example: 

```bash
python LlavaCode_inference.py
```

This script reads a test example from file test_example.jsonl and performs an inference and **EM**/**ES** calculations. 
You can adjust parameters by changing the attributes of ArgsMock class in the beginning of the script. 

For example, change the number of projected cross-file contexts appended to the line completion prompt: 
```bash
num_structure_tokens = 10
```

num_structure_tokens = 0 preformes inference without cross-file context augmentation.

## Running the Benchmarks

First, make the script you want to run executable:

```bash
chmod +x script.sh
```

To evaluate the Python model on the Repoeval benchmark, run:

```bash
./run_python_benchmark.sh
```

To evaluate the Java model on the Repoeval benchmark, run:

```bash
./run_java_benchmark.sh
```

To evaluate the Java model on the subset of the test TheStack2 dataset, run:

```bash
./test_java_stack2.sh
```

All scripts evaluate **EM** (Exact Match) and **ES** (Edit Similarity) metrics on the provided dataset.
