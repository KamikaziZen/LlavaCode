# LlavaCode

## Environment Setup

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Data Preparation

Unzip the datasets archives into your working directory.

```stack_java_test_1k``` - subset of 1k samples randomly chosen from the test split of TheStackv2 dataset.
```python_line_completion_sparse_bm25.jsonl``` - repoeval benchmark (as described in RepoCoder paper)

## Checkpoint Preparation

Unzip the provided checkpoints archive and place the `.ckpt` files in the ./ckpt directory.

The expected directory structure is:

```
./stack2_java_test_1k
./python_line_completion_sparse_bm25.jsonl
./ckpt/qwen7_unixcoder_4l_java_2emes_stack2.ckpt
./ckpt/qwen7_unixcoder_3l_python_emes_stack.ckpt
./test_java.sh
./test_python.sh
...
```


## Running the Benchmark

To evaluate the java model on the provided subset of the test dataset, run:

```bash
./test_java.sh
```

If necessary, make the script executable first:

```bash
chmod +x test_java.sh
```

To evaluate the python model on the repoeval benchmark of the dataset, run:

```bash
./test_python.sh
```

If necessary, make the script executable first:

```bash
chmod +x test_python.sh
```

The script loads the checkpoint and evaluates **EM** (Exact Match) and **ES** (Edit Similarity) metrics on the dataset subset.
