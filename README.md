# DP-vllm-Inference-superfast

이 코드는 데이터 병렬(Data Parallel, DP) 방식으로 여러 GPU를 동시에 사용해, 한 장의 GPU 메모리에 적재 가능한 크기의 단일 모델(예: Qwen/Qwen3-30B-A3B-Instruct-2507)을 기반으로 대규모 JSONL 데이터에 대해 빠르게 생성/증강 작업을 수행하기 위해 작성되었습니다. 따라서, 모델은 단일 GPU에 올라가는 크기임을 가정합니다. 

### 동작 개요

1. 입력 JSONL을 GPU 개수만큼 균등 샤딩 → 각 샤드를 개별 프로세스(+GPU)에 배정

2. CHAT_FIELD(예: "chat")를 토크나이저 chat template로 변환 후 vLLM에 배치로 입력

3. 생성 결과를 원본 데이터에 GEN_FIELD를 추가하여 저장

- 부분 파일: OUT_DIR/FILE_NAME.part{0..N-1}.jsonl

- 체크포인트: OUT_DIR/FILE_NAME.part{idx}.ckpt.json

- 최종 병합: OUT_DIR/FILE_NAME.jsonl

### 가이드

**MAX_MODEL_LEN**: 입력+출력의 합 상한

vLLM은 모델의 최대 컨텍스트 길이를 기본으로 잡습니다(대개 매우 큼).

작업에 필요한 수준(8k~16k 등)으로 명시해야 KV 캐시 메모리 낭비를 막습니다.

**MAX_NEW_TOKENS**: 실제 필요한 출력 길이에 맞춰 설정

### Trouble Shooting
**- 엔진 초기화 실패/메모리 부족(OOM)**

MAX_MODEL_LEN↓, CHUNK_SIZE↓, MAX_NEW_TOKENS↓
    
**- 너무 느려요**

CHUNK_SIZE를 단계적으로↑
    
I/O가 병목이면 저장 주기(SAVE_EVERY) 조정

**- 출력이 중간에 끊겨요**
    
MAX_NEW_TOKENS↑

### TODO
1. reasoning hybrid 모델 사용 시 on/off
2. args

