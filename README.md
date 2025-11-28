## About
This branch forks the excellent epub-translator https://github.com/oomol-lab/epub-translator and simplifies the logic of the prompts. As a result, this branch is expected to be with:

 * More compatibility with the local median size model like Phi4
 * Lower token consumption to 1/3
 * Less failed LLM requests
 * More reasonable output layout

To use this branch, download the zipped source codes and ```pip install ***.zip```

``` python
llm = LLM(
  key="ttest", # LLM's API key
  #url="http://localhost:8080/v1", # LLM's base URL
  model="phi4", # LLM's model name
  token_encoding="o200k_base", # Local model for calculating the number of tokens
  temperature=0.8,
  top_p=0.95,
    timeout=250.0,
    retry_times=5,
    retry_interval_seconds=6.0,
)

from tqdm import tqdm
with tqdm(total=1.0, desc="Translating",ncols=80 ) as bar:
  def refresh_progress(progress: float) -> None:
    bar.n = progress
    bar.refresh()

  translate(
    llm=llm, # llm object constructed in the previous step
    source_path="R:/FUN/trans/mo.epub", # Original EPUB file to be translated
    translated_path="R:/FUN/trans/mo_out.epub", # Path to save the translated EPUB
    target_language=Language.SIMPLIFIED_CHINESE, # Target language for translation, in this case English.
    working_path="R:/FUN/trans",
    report_progress=refresh_progress,
    max_chunk_tokens_count=768, # No need to be high as this will increase the time for retry
    max_threads_count= 1,
  )
```
