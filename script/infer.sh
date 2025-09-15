MODEL_DIR=$1

echo "Testing MMAR with model: ${MODEL_DIR}"
OUT_DIR="$MODEL_DIR/grpo_MMAR/"
mkdir -p ${OUT_DIR}
TEST_FILE=data/MMAR/MMAR-meta.json
python data/MMAR/test.py --model_path ${MODEL_DIR} --batch_size 8 --data_file ${TEST_FILE} --out_file ${OUT_DIR}/mmar-meta.jsonl --think True --think_max_len 100 --random_drop True
python data/MMAR/evaluation.py --input ${OUT_DIR}/mmar-meta.jsonl > ${OUT_DIR}/mmar-meta-eval.json
echo "Completed test. Generated done file: ${OUT_DIR}/mmar-meta-eval.json"


echo "Testing MMAU with model: ${MODEL_DIR}"
OUT_DIR="$MODEL_DIR/grpo_MMAU/"
mkdir -p ${OUT_DIR}
TEST_FILE=data/MMAU/mmau-test-mini.json
python data/MMAU/test.py --model_path ${MODEL_DIR} --batch_size 8 --data_file ${TEST_FILE} --out_file ${OUT_DIR}/mmau-test-mini.jsonl --think True --think_max_len 100 --random_drop True
python data/MMAU/evaluation.py --input ${OUT_DIR}/mmau-test-mini.jsonl > ${OUT_DIR}/mmau-test-mini-eval.json
echo "Completed test. Generated done file: ${OUT_DIR}/mmau-test-mini-eval.json"


TEST_FILE=data/MMAU/mmau-test.json
python data/MMAU/test.py --model_path ${MODEL_DIR} --batch_size 8 --data_file ${TEST_FILE} --out_file ${OUT_DIR}/mmau-test.jsonl --think True --think_max_len 100 --random_drop True
python data/MMAU/evaluation.py --input ${OUT_DIR}/mmau-test.jsonl > ${OUT_DIR}/mmau-test-eval.json
echo "Completed test. Generated done file: ${OUT_DIR}/mmau-test-eval.json"
