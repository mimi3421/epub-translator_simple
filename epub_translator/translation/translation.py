from math import ceil
from typing import Callable, Iterator, Generator
from pathlib import Path
from concurrent.futures import as_completed, ThreadPoolExecutor
from threading import Lock
from xml.etree.ElementTree import Element
import re

from ..llm import LLM
from ..xml import encode_friendly

from .types import language_chinese_name, Fragment, Language
from .store import Store
from .splitter import split_into_chunks
from .chunk import match_fragments, Chunk
from .utils import is_empty, clean_spaces


ProgressReporter = Callable[[float], None]

lexi_dict = {}
lexi_dict_lock = Lock()

def translate(
      llm: LLM,
      gen_fragments_iter: Callable[[], Iterator[Fragment]],
      cache_path: Path | None,
      target_language: Language,
      user_prompt: str | None,
      max_chunk_tokens_count: int,
      max_threads_count: int,
      report_progress: ProgressReporter,
    ) -> Generator[str, None, None]:

  if user_prompt is not None:
    user_prompt = _normalize_user_input(user_prompt.splitlines())

  store = Store(cache_path) if cache_path else None
  chunk_ranges = list(split_into_chunks(
    llm=llm,
    fragments_iter=gen_fragments_iter(),
    max_chunk_tokens_count=max_chunk_tokens_count,
  ))
  with ThreadPoolExecutor(max_workers=max_threads_count) as executor:
    futures = [
      executor.submit(lambda chunk=chunk: (chunk, _translate_chunk(
        llm=llm,
        store=store,
        chunk=chunk,
        target_language=target_language,
        user_prompt=user_prompt,
      )))
      for chunk in match_fragments(
        llm=llm,
        target_language=target_language,
        chunk_ranges_iter=iter(chunk_ranges),
        fragments_iter=gen_fragments_iter(),
      )
    ]
    def _generate_chunks_from_futures():
      try:
        for future in as_completed(futures):
          yield future.result()
      except Exception as err:
        for future in futures:
          if not future.done():
            future.cancel()
        raise err

    yield from _sort_translated_texts_by_chunk(
      target=_generate_chunks_from_futures(),
      total_tokens_count=sum(chunk.tokens_count for chunk in chunk_ranges),
      report_progress=report_progress,
    )

def _sort_translated_texts_by_chunk(
      target: Iterator[tuple[Chunk, list[str]]],
      total_tokens_count: int,
      report_progress: ProgressReporter,
    ) -> Generator[str, None, None]:

  buffer: list[tuple[Chunk, list[str]]] = []
  wanna_next_index: int = 0
  translated_tokens_count: int = 0

  for chunk, translated_texts in target:
    buffer.append((chunk, translated_texts))
    if wanna_next_index == chunk.index:
      buffer.sort(key=lambda e: e[0].index)
      to_clear: list[list[str]] = []

      for chunk, translated_texts in buffer:
        if chunk.index > wanna_next_index:
          break
        to_clear.append(translated_texts)
        if chunk.index == wanna_next_index:
          wanna_next_index += 1

      if to_clear:
        buffer = buffer[len(to_clear):]
        for translated_texts in to_clear:
          yield from translated_texts

    translated_tokens_count += chunk.tokens_count
    report_progress(float(translated_tokens_count) / total_tokens_count)

def _translate_chunk(
      llm: LLM,
      store: Store | None,
      chunk: Chunk,
      target_language: Language,
      user_prompt: str | None,
    ) -> list[str]:

  translated_texts: list[str] | None = None
  source_texts = chunk.head + chunk.body + chunk.tail
  if store is not None:
    translated_texts = store.get(chunk.hash)
    if translated_texts is not None and \
        len(source_texts) != len(translated_texts):
      translated_texts = None
      print(f"Warning: Mismatched lengths in cached translation for chunk: {chunk.hash.hex()}",)

  if translated_texts is None:
    translated_texts = [
      clean_spaces(text)
      for text in _translate_texts(
        llm=llm,
        texts=source_texts,
        texts_tokens=chunk.tokens_count,
        target_language=target_language,
        user_prompt=user_prompt,
      )
    ]
    if store is not None:
      store.put(chunk.hash, translated_texts)

  head_length = len(chunk.head)
  translated_texts = translated_texts[head_length:head_length + len(chunk.body)]

  return translated_texts

_PLAIN_TEXT_SCALE = 2.0
_XML_TEXT_SCALE = 2.5

def _translate_texts(
      llm: LLM,
      texts: list[str],
      texts_tokens: int,
      target_language: Language,
      user_prompt: str | None,
    ) -> list[str]:

  original_text = _normalize_user_input(texts)
  if original_text is None:
    return [""] * len(texts)
    
  # create text map
  original_text_map = []
  user_data_dict = []
  # 翻转序列以免语义连续的片段被翻译在一起
  # 注意在输出结果时再返回来
  for t in reversed(texts): 
    #if t.strip() == '':
    if re.fullmatch(r'^[^\w]*$',t) or re.fullmatch(r'^[\W\d]*$',t): # do not translate the lines with only blanks and number
      original_text_map.append(t)
    #elif len(t)<6: # do not translate the short lines
    #  original_text_map.append(t)
    else:
      user_data_dict.append(t.strip())
      original_text_map.append(f"__trans_id__{str(len(user_data_dict)-1)}")
      
  # add a definited normal last line to ensure all short phrases at the end are translated.
  # No need to remove the line as it is not shown in the original_text_map
  user_data_dict += ['By the way, John and Rose are couples.']
  
  user_data_dict = {str(i) : t.replace("\t"," ") for i,t in enumerate(user_data_dict)}
  user_data = '\n'.join([f"{i}\t{t}" for i,t in user_data_dict.items()])+'\n'
  user_data = f"\n``` tsv\n{user_data}\n```\n"

  # generate lexi
  global lexi_dict
  with lexi_dict_lock :
    # Initiate the lexi_dict
    #if not lexi_dict:
    #  lexi_dict['the mom test'] = '妈妈测试'
      
    lexi_string = ''
    # get the lexi match counts
    all_lexi_count = {k : l for k,x in lexi_dict.items() if ( l:=len(re.findall(re.sub(r'\\(?!n)',r'\\\\',k), user_data.lower(), flags=0))) > 0 }
    if len(all_lexi_count) > 0:
      # only export the top 20 lexi
      the_top = sorted(all_lexi_count.values(),reverse = True)[0:20][-1] # the count at the top 20
      lexi_string = '\n'.join([f'"{k}":"{lexi_dict[k]}"' for k,v in all_lexi_count.items() if v >= the_top])
      lexi_string = f'''
  参考译文：
  {lexi_string}
  '''
  #user_data = original_text
  user_data = f"{lexi_string}\n\n{user_data}"
  
  if user_prompt is not None:
    user_data = f"其他翻译要求：\n{user_prompt}\n\n{user_data}"
    
  translated_text = llm.request_text_JSON(
    template_name="translate",
    text_tag="TXT",
    user_data=user_data,
    #parser=lambda r: r,
    parser=lambda r: _parse_translated_response_JSON(r,user_data_dict,original_text_map),
    max_tokens=ceil(texts_tokens * _PLAIN_TEXT_SCALE),
    params={
      "target_language": language_chinese_name(target_language),
      "user_prompt": user_prompt,
    },
  )

  return reversed(translated_text)

def _parse_translated_response_JSON(res_par : dict, user_data_dict : dict, original_text_map : list[str]) -> list[str]:
  if not ('data' in res_par.keys() and 'lexi' in res_par.keys()) :
    raise ValueError(f"JSON parse error: data and lext not exist.\n{str(res_par)}")
    
  # deal with data
  if len(res_par['data'].keys() ^ user_data_dict.keys()) != 0: # not all key match, some texts missed
      raise ValueError(f"JSON parse error: Text ID not match. Some texts missed.\n{str(res_par)}\n{str(user_data_dict)}\n{str(original_text_map)}")
        
  # deal with lexi
  try:
    global lexi_dict
    for k,x in res_par['lexi'].items():
      #if k == x or re.fullmatch(r'^[^\w]*$',k) or re.fullmatch(r'^[^\w]*$',x):
      if re.fullmatch(r'^[^\w]*$',k) or re.fullmatch(r'^[^\w]*$',x):
        # The lexi didnt' translate anything
        continue
      k = k.strip().lower()
      with lexi_dict_lock:
        if k not in lexi_dict.keys():
            lexi_dict[k] = x.strip()
  except:
    # just do nothing if the lexi parse error
    pass
        
  # output the translated texts or the blank texts
  return [ res_par['data'][t] if (t[:12]=='__trans_id__' and t[12:] in res_par['data'].keys()) else t for t in original_text_map]
  
def _parse_translated_response(resp_element: Element, sources_count: int) -> list[str]:
  fragments: list[str | None] = [None] * sources_count
  for fragment_element in resp_element:
    if fragment_element.text is None:
      continue
    id = fragment_element.get("id", None)
    if id is None:
      continue
    index = int(id) - 1
    if index < 0 or index >= len(fragments):
      raise ValueError(f"invalid fragment id: {id}")
    fragments[index] = fragment_element.text.strip()

  # 有时 LLM 会将多段融合在一起，这里尽可能让译文靠后，将空白段留在前面。
  # 这样看起来一大段的译文对应若干小段原文，观感更好。
  for i in range(len(fragments)):
    fragment = fragments[i]
    if fragment is not None and i < len(fragments) - 1:
      next_fragment = fragments[i + 1]
      if next_fragment is None:
        fragments[i] = None
        fragments[i + 1] = fragment

  return [f or "" for f in fragments]

def _normalize_user_input(user_lines: list[str]) -> str | None:
  empty_lines_count: int = 0
  lines: list[str] = []
  for line in user_lines:
    if is_empty(line):
      empty_lines_count += 1
    else:
      if lines:
        if empty_lines_count >= 2:
          lines.append("")
          lines.append("")
        elif empty_lines_count == 1:
          lines.append("")
      lines.append(clean_spaces(line))
  if not lines:
    return None
  return "\n".join(lines)
