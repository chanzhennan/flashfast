#!/bin/bash

benchmark=$1
warmup=$2
rep=$3

python_scripts=(
    "decode_mha_test.py"                    # MHA 测试
    "decode_mha_alibi_masked_test.py"       # MHA 测试（Alibi Masked）
    "decode_mqa_test.py"                    # MQA 测试
    "decode_mqa_t_test.py"                  # MQA 测试（shape [seqlen, bs, ...]）
    "decode_serving_mha_test.py"            # MHA 测试（serving 模式）
)

isprefills=(
    ""                                      # decode 测试
    "--is_prefill"                          # prefill 测试
)

# 迭代遍历 Python 脚本命令列表
for python_script in "${python_scripts[@]}"
do
    for isprefill in "${isprefills[@]}"
    do
        script_command="python $python_script $isprefill --warmup $warmup --rep $rep"
        if [ "$benchmark" == "true" ]; then
            script_command="$script_command --benchmark"
        fi

        # 输出当前调用的命令行输入
        echo "Running: $script_command"

        # 执行 Python 脚本
        eval "$script_command"

        # 检查 Python 脚本的退出状态
        if [ $? -eq 0 ]; then
            echo "Success"
        else
            echo "Error: Python script failed"
            exit 1
        fi

        echo "---------------------"
    done
done
