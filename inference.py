import os
import json
from math import ceil
from multiprocessing import Process
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_PATH = "MODEL_NAME"
INPUT_JSONL = "INPUT_JSONL_PATH"  
OUT_DIR = "OUT_DIR_PATH"
FILE_NAME = "FILE_NAME"  # 예: "tagged_data"
MERGED_OUT = os.path.join(OUT_DIR, FILE_NAME+".jsonl")

GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]  # 사용 GPU 목록
CHUNK_SIZE = 5000       # vLLM 배치 제출 크기(프롬프트 수)
SAVE_EVERY = 200        # 이 개수마다 강제 저장/체크포인트

CHAT_FIELD = "CHAT_FIELD"  # 데이터에서 채팅 프롬프트 필드 이름 예: "chat"
GEN_FIELD = "GEN_FIELD"    # 데이터에서 생성된 텍스트 필드 이름 예: "tagged"

MAX_NEW_TOKENS = 384
MAX_MODEL_LEN = 8192      # 프롬프트+출력 합 상한         

def load_data(path: str) -> List[Dict[str, Any]]:
    '''
    JSONL 파일을 읽어 각 줄을 JSON으로 파싱해 리스트로 반환

    Args:
        path (str): 입력 JSONL 파일 경로

    Returns:
        List[Dict[str, Any]]: 각 라인을 파싱한 딕셔너리의 리스트
    '''
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def write_line_safely(fh, obj: Dict[str, Any]):
    '''
    객체를 JSON 문자열로 직렬화, 한 줄로 파일에 기록

    Args:
        fh: 쓰기 모드
        obj (Dict[str, Any]): 직렬화하여 기록할 객체
    '''
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    fh.write(line)

def flush_and_fsync(fh):
    '''
    파일 버퍼를 플러시하고 fsync로 디스크에 강제 반영

    Args:
        fh: 열린 파일 핸들
    '''
    fh.flush()
    os.fsync(fh.fileno())

def save_ckpt(ckpt_path: str, processed: int):
    '''
    진행 상황을 체크포인트 파일( JSON )로 저장

    Args:
        ckpt_path (str): 체크포인트 파일 경로
        processed (int): 지금까지 처리한 샘플 수
    '''
    with open(ckpt_path, "w", encoding="utf-8") as cf:
        json.dump({"processed": processed}, cf)

def load_ckpt(ckpt_path: str) -> int:
    '''
    체크포인트 파일을 읽어 처리된 샘플 수를 반환
    파일이 없거나 파싱 실패 시 0을 반환

    Args:
        ckpt_path (str): 체크포인트 파일 경로

    Returns:
        int: 처리된 샘플 수(없으면 0)
    '''
    if not os.path.exists(ckpt_path):
        return 0
    try:
        with open(ckpt_path, "r", encoding="utf-8") as cf:
            obj = json.load(cf)
            return int(obj.get("processed", 0))
    except Exception:
        return 0

def worker(gpu_id: int, shard_idx: int, shard: List[Dict[str, Any]], tmp_out_path: str, ckpt_path: str):
    '''
    단일 GPU에서 샤드 데이터를 생성하는 워커 프로세스
    - 토크나이저/LLM 로드
    - 샤드 데이터를 CHUNK_SIZE 단위로 배치 생성
    - 생성 결과를 GEN_FIELD 키에 저장하여 임시 jsonl로 기록
    - SAVE_EVERY마다 체크포인트 및 fsync

    Args:
        gpu_id (int): 사용할 GPU ID (CUDA_VISIBLE_DEVICES 설정에 사용)
        shard_idx (int): 샤드 인덱스(로그용)
        shard (List[Dict[str, Any]]): 이 워커가 담당할 데이터 샤드
        tmp_out_path (str): 임시 출력 jsonl 경로
        ckpt_path (str): 체크포인트 파일 경로
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenized_prompts = [
        tokenizer.apply_chat_template(d[CHAT_FIELD], tokenize=False, add_generation_prompt=True)
        for d in shard
    ]

    already = load_ckpt(ckpt_path)
    already = max(0, min(already, len(tokenized_prompts)))

    llm = LLM(model=MODEL_PATH, 
              tensor_parallel_size=1,
              max_model_len=MAX_MODEL_LEN,)
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        max_tokens=MAX_NEW_TOKENS,
        skip_special_tokens=False,
    )

    mode = "a" if os.path.exists(tmp_out_path) else "w"
    with open(tmp_out_path, mode, encoding="utf-8") as fout:
        processed = already 
        since_flush = 0

        total = len(tokenized_prompts)
        for s in range(already, total, CHUNK_SIZE):
            e = min(s + CHUNK_SIZE, total)
            batch_prompts = tokenized_prompts[s:e]
            batch_raw = shard[s:e]

            try:
                outputs = llm.generate(batch_prompts, sampling_params)
            except Exception as ex:
                save_ckpt(ckpt_path, processed)
                flush_and_fsync(fout)
                raise ex

            for i, out in enumerate(outputs):
                try:
                    token_ids = out.outputs[0].token_ids[:-1]
                    gen_text = tokenizer.decode(token_ids, skip_special_tokens=False)

                    curr = batch_raw[i].copy()
                    curr[GEN_FIELD] = gen_text
                    write_line_safely(fout, curr)

                    processed += 1
                    since_flush += 1

                    if since_flush >= SAVE_EVERY:
                        save_ckpt(ckpt_path, processed)
                        flush_and_fsync(fout)
                        since_flush = 0
                except Exception:
                    continue

        save_ckpt(ckpt_path, processed)
        flush_and_fsync(fout)

    print(f"[GPU {gpu_id}] shard {shard_idx} done -> {tmp_out_path} (processed={processed}/{len(shard)})")

def merge_parts(part_paths: List[str], merged_path: str):
    '''
    여러 파트 jsonl 파일을 순서대로 읽어 하나의 jsonl 파일로 병합

    Args:
        part_paths (List[str]): 파트 파일 경로 리스트
        merged_path (str): 병합 결과 파일 경로
    '''
    with open(merged_path, "w", encoding="utf-8") as fout:
        for p in part_paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)

def main():
    '''
    전체 파이프라인을 실행
    - 출력 디렉터리 생성
    - 입력 JSONL 로드 및 GPU 수만큼 균등 샤딩
    - 각 샤드에 대해 프로세스 생성/실행
    - 완료 후 파트 파일들을 병합하여 최종 결과 생성
    '''
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_data(INPUT_JSONL)

    n = len(GPU_IDS)
    per_shard = ceil(len(data) / n)

    procs = []
    part_paths = []
    ckpt_paths = []

    for shard_idx, gpu_id in enumerate(GPU_IDS):
        start = shard_idx * per_shard
        end = min((shard_idx + 1) * per_shard, len(data))
        if start >= end:
            break

        shard = data[start:end]
        tmp_out = os.path.join(OUT_DIR, f"{FILE_NAME}.part{shard_idx}.jsonl")
        ckpt_path = os.path.join(OUT_DIR, f"{FILE_NAME}.part{shard_idx}.ckpt.json")

        part_paths.append(tmp_out)
        ckpt_paths.append(ckpt_path)

        p = Process(target=worker, args=(gpu_id, shard_idx, shard, tmp_out, ckpt_path))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    merge_parts(part_paths, MERGED_OUT)
    print(f"[merge] -> {MERGED_OUT}")

if __name__ == "__main__":
    main()
