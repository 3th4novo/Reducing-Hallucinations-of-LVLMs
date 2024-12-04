python ./eval/object_hallucination_vqa_llava.py \
--model-path "liuhaotian/llava-v1.5-7b" \
--question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./output/llava15_${dataset_name}_pope_${type}_answers_no_cd_seed${seed}.jsonl