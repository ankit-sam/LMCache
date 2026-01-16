Local storage with io_uring
===========================

.. _local-uring-storage-overview:

Overview
--------

CPU RAM and Local Storage are the two ways of offloading KV cache onto non-GPU
memory of the same machine that is running inference.

LMCache provides two ways of offloading KV Cache to local disk. By default it utilizes
the exisitng python operating system write calls on a given file descriptor.

It can also utilize liburing to offload and retrieve the KV cache from local disk.

.. _local-uring-storage-online-inference-example:

Online Inference Example
------------------------

This example is built based on the :doc:`LOCAL STORAGE <./local_storage>` example.
This is also meant for comparing the exisitng disk offload and retrieve throughput
with the io_uring backend throughput.

Let's see the TTFT (time to first token) and E2EL (Total latency)

.. _local-uring-storage-prerequisites:

**Prerequisites:**

- A Machine with at least one GPU. Adjust the max model length of your vllm instance depending on your GPU memory and the long context you want to use.

- vllm and lmcache installed (:doc:`Installation Guide <../../getting_started/installation>`)

- A few packages:

.. code-block:: bash

    pip install openai transformers liburing



**Step 0. Set up a directory for this example:**

.. code-block:: bash

    mkdir lmcache-local-disk-example
    cd lmcache-local-disk-example

**Step 1. Prepare a long context!**

We want a context long enough that vllm's prefix caching will not be able to hold the KV caches in
GPU memory and LMCache is necessary to keep KV caches in non-GPU memory:

.. code-block:: bash

    man bash > man-bash.txt
    man git > man-git.txt
    man fio > man-fio.txt

**Step 2. Start a vLLM server with Disk offloading enabled:**

Create a an lmcache configuration file called: ``disk-offload.yaml``

Example ``disk-offload.yaml``:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: false
    max_local_cpu_size: 5.0
    enable_async_loading: true
    local_disk: "file:///local/disk_test/local_disk/"
    max_local_disk_size: 5.0

    extra_config:
        use_odirect: True
        use_uring: true
        ring_size: 4096
        max_batch: 256

.. code-block:: bash

    LMCACHE_CONFIG_FILE="disk-offload.yaml" \
    vllm serve \
        Qwen/Qwen3-8B \
        --gpu-memory-utilization 0.5 \
        --max-model-len 16384 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

**Step 3. Query for TTFT and total latency with LMCache:**

Once the Open AI compatible server is running on default vllm port 8000, let's query it!

Create a script called ``query-long.py`` and paste the following code:

.. code-block:: python

    import time
    import sys
    from openai import OpenAI
    from transformers import AutoTokenizer

    client = OpenAI(
        api_key="dummy-key",  # required by OpenAI client even for local servers
        base_url="http://localhost:8000/v1"
    )

    models = client.models.list()
    model = models.data[0].id

    filename = sys.argv[1]
    long_context = ""
    with open(filename, "r") as f:
        long_context = f.read()

    # a truncation of the long context for the --max-model-len 16384
    # if you increase the --max-model-len, you can decrease the truncation i.e.
    # use more of the long context
    long_context = long_context[:50000]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    question = "Summarize {filename} in 2 sentences."

    prompt = f"{long_context}\n\n{question}"

    print(f"Number of tokens in prompt: {len(tokenizer.encode(prompt))}")

    def query_and_measure_ttft():
        start = time.perf_counter()
        ttft = None
        e2el = None

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7,
            stream=True,
        )

        for chunk in chat_completion:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                if ttft is None:
                    ttft = time.perf_counter()
                print(chunk_message, end="", flush=True)

        e2el = time.perf_counter()
        print("\n")  # New line after streaming
        return (ttft - start), (e2el - start)

    print("Querying vLLM server with LMCache Disk Offload")
    ttft, e2el = query_and_measure_ttft()
    print(f"TTFT: {ttft:.3f} seconds, Total Latency: {e2el:.3f} seconds")

Then run:

.. code-block:: bash

    python query-long.py man-git.txt


**Example Output:**


.. code-block:: text

    Number of tokens in prompt: 10476

    ...
    The user wants a two-sentence summary. The first sentence should introduce Git and
    its main features. The second sentence should cover the structure of the manual and
    key components like commands, configuration, and environment variables. I need to make
    sure it's concise and captures the essence without getting too technical.

    The Git manual is a comprehensive guide to using Git, a distributed version control
    system, covering commands, configuration, and workflows for managing source code.
    It details high-level (porcelain) and low-level (plumbing) commands, configuration
    mechanisms, environment variables, and core concepts like repositories, objects, and branching.

    TTFT: 3.616 seconds, Total Latency: 6.804 seconds


.. _local-storage-uring-consideration:

Considerations:
-----

- To trigger read operations run the ``query-long.py`` script multiple times with different contexts.
  This depends on the model weight and gpu memory utilization. Below code ideally should be able to
  trigger async load from the local disk.

.. code-block:: yaml

   python query-long.py man-git.txt
   python query-long.py man-bash.txt
   python query-long.py man-fio.txt
   python query-long.py man-git.txt
