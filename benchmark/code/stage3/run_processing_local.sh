#!/bin/bash

# MIMIC-CXR-VQA 样本批量处理启动脚本 - 本地VLLM版本

echo "=== MIMIC-CXR-VQA 样本批量处理（本地VLLM版本）==="
echo "总数据量: 约7400条（train: 6660, test: 741）"
echo ""

# 激活conda环境
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ms-swift}"
echo "激活${CONDA_ENV_NAME}环境..."
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
elif [ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV_NAME"
else
    echo "警告：未找到conda命令；将使用当前shell环境继续"
fi

# 设置代理访问Hugging Face
# echo "设置代理..."
# export https_proxy=http://127.0.0.1:7890
# export http_proxy=http://127.0.0.1:7890
# echo "代理设置完成: $http_proxy"

# 设置GPU设备（使用A6000显卡索引1,2）
export CUDA_VISIBLE_DEVICES=1,2,3,4
echo "设置GPU设备: $CUDA_VISIBLE_DEVICES"

# 配置参数
VLLM_SERVER_URL="http://localhost:8000"
VLLM_MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
VLLM_HOST="0.0.0.0"
VLLM_PORT="8000"
TENSOR_PARALLEL_SIZE=4  # 使用2张A6000显卡

# 检查必要文件是否存在
if [ ! -f "train_samples.json" ]; then
    echo "错误：找不到 train_samples.json"
    exit 1
fi

if [ ! -f "test_samples.json" ]; then
    echo "错误：找不到 test_samples.json"
    exit 1
fi

if [ ! -f "cxr_grade_er_conversion_final_20250712_200006.json" ]; then
    echo "错误：找不到 CXR规则文件"
    exit 1
fi

if [ ! -f "optimized_batch_processor_local.py" ]; then
    echo "错误：找不到本地批量处理脚本"
    exit 1
fi

if [ ! -f "smart_rule_matcher_local.py" ]; then
    echo "错误：找不到本地智能规则匹配器"
    exit 1
fi

# 检查vllm命令是否可用
if ! command -v vllm &> /dev/null; then
    echo "错误：找不到vllm命令，请确保已在ms-swift环境中安装VLLM"
    exit 1
fi

# 检查VLLM服务器状态的函数
check_vllm_server() {
    echo "检查VLLM服务器状态..."
    response=$(curl -s -o /dev/null -w "%{http_code}" "$VLLM_SERVER_URL/health" --connect-timeout 5)
    if [ "$response" = "200" ]; then
        echo "✅ VLLM服务器运行正常"
        return 0
    else
        echo "❌ VLLM服务器未响应或异常"
        return 1
    fi
}

# 启动VLLM服务器的函数
start_vllm_server() {
    echo "正在启动VLLM服务器..."
    echo "模型: $VLLM_MODEL"
    echo "地址: $VLLM_HOST:$VLLM_PORT"
    echo "GPU设备: $CUDA_VISIBLE_DEVICES"
    echo "张量并行大小: $TENSOR_PARALLEL_SIZE"
    echo ""
    echo "注意：首次启动可能需要下载模型，请耐心等待..."
    echo ""
    
    # 在后台启动VLLM服务器（使用标准vllm serve命令）
    nohup vllm serve "$VLLM_MODEL" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len 32768 \
        --trust-remote-code \
        > vllm_server.log 2>&1 &
    
    VLLM_PID=$!
    echo "VLLM服务器已启动，PID: $VLLM_PID"
    echo $VLLM_PID > vllm_server.pid
    
    echo "等待服务器初始化..."
    for i in {1..60}; do
        sleep 5
        if check_vllm_server; then
            echo "服务器启动成功！"
            return 0
        fi
        echo "等待中... (${i}/60)"
    done
    
    echo "服务器启动超时，请检查日志文件 vllm_server.log"
    return 1
}

# 停止VLLM服务器的函数
stop_vllm_server() {
    if [ -f "vllm_server.pid" ]; then
        VLLM_PID=$(cat vllm_server.pid)
        echo "正在停止VLLM服务器 (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null
        sleep 3
        
        # 强制杀死进程
        if kill -0 $VLLM_PID 2>/dev/null; then
            echo "强制停止VLLM服务器..."
            kill -9 $VLLM_PID 2>/dev/null
        fi
        
        rm -f vllm_server.pid
        echo "VLLM服务器已停止"
    else
        echo "未找到VLLM服务器PID文件"
        # 尝试根据端口杀死进程
        VLLM_PID=$(lsof -ti:$VLLM_PORT 2>/dev/null)
        if [ ! -z "$VLLM_PID" ]; then
            echo "发现端口$VLLM_PORT上的进程 (PID: $VLLM_PID)，正在停止..."
            kill $VLLM_PID 2>/dev/null
            sleep 3
            if kill -0 $VLLM_PID 2>/dev/null; then
                kill -9 $VLLM_PID 2>/dev/null
            fi
            echo "进程已停止"
        fi
    fi
}

# 提供选项菜单
echo "请选择操作："
echo "1. 启动VLLM服务器并处理训练集 (train_samples.json -> train_processed_local.json)"
echo "2. 启动VLLM服务器并处理测试集 (test_samples.json -> test_processed_local.json)"
echo "3. 启动VLLM服务器并处理所有数据"
echo "4. 验证已处理的文件"
echo "5. 从中断点恢复训练集处理"
echo "6. 从中断点恢复测试集处理"
echo "7. 仅启动VLLM服务器"
echo "8. 仅停止VLLM服务器"
echo "9. 检查VLLM服务器状态"
echo ""

read -p "请输入选择 (1-9): " choice

case $choice in
    1)
        echo "启动VLLM服务器并处理训练集..."
        if ! check_vllm_server; then
            start_vllm_server
            if [ $? -ne 0 ]; then
                echo "VLLM服务器启动失败，退出"
                exit 1
            fi
        fi
        
        echo "开始处理训练集..."
        python3 optimized_batch_processor_local.py \
            --input train_samples.json \
            --output train_processed_local.json \
            --server-url "$VLLM_SERVER_URL" \
            --checkpoint-interval 50
        ;;
    2)
        echo "启动VLLM服务器并处理测试集..."
        if ! check_vllm_server; then
            start_vllm_server
            if [ $? -ne 0 ]; then
                echo "VLLM服务器启动失败，退出"
                exit 1
            fi
        fi
        
        echo "开始处理测试集..."
        python3 optimized_batch_processor_local.py \
            --input test_samples.json \
            --output test_processed_local.json \
            --server-url "$VLLM_SERVER_URL" \
            --checkpoint-interval 25
        ;;
    3)
        echo "启动VLLM服务器并处理所有数据..."
        if ! check_vllm_server; then
            start_vllm_server
            if [ $? -ne 0 ]; then
                echo "VLLM服务器启动失败，退出"
                exit 1
            fi
        fi
        
        echo "处理训练集..."
        python3 optimized_batch_processor_local.py \
            --input train_samples.json \
            --output train_processed_local.json \
            --server-url "$VLLM_SERVER_URL" \
            --checkpoint-interval 50
        
        if [ $? -eq 0 ]; then
            echo "训练集处理完成，开始处理测试集..."
            python3 optimized_batch_processor_local.py \
                --input test_samples.json \
                --output test_processed_local.json \
                --server-url "$VLLM_SERVER_URL" \
                --checkpoint-interval 25
        else
            echo "训练集处理失败，停止"
            exit 1
        fi
        ;;
    4)
        echo "验证处理结果..."
        if [ -f "train_processed_local.json" ]; then
            echo "验证训练集结果："
            python3 optimized_batch_processor_local.py \
                --output train_processed_local.json \
                --server-url "$VLLM_SERVER_URL" \
                --validate --analyze
        fi
        
        if [ -f "test_processed_local.json" ]; then
            echo "验证测试集结果："
            python3 optimized_batch_processor_local.py \
                --output test_processed_local.json \
                --server-url "$VLLM_SERVER_URL" \
                --validate --analyze
        fi
        ;;
    5)
        echo "从中断点恢复训练集处理..."
        if ! check_vllm_server; then
            start_vllm_server
            if [ $? -ne 0 ]; then
                echo "VLLM服务器启动失败，退出"
                exit 1
            fi
        fi
        
        if [ -f "train_processed_local_checkpoint.json" ]; then
            python3 optimized_batch_processor_local.py \
                --input train_samples.json \
                --output train_processed_local.json \
                --server-url "$VLLM_SERVER_URL" \
                --checkpoint-interval 50
        else
            echo "未找到训练集检查点文件"
        fi
        ;;
    6)
        echo "从中断点恢复测试集处理..."
        if ! check_vllm_server; then
            start_vllm_server
            if [ $? -ne 0 ]; then
                echo "VLLM服务器启动失败，退出"
                exit 1
            fi
        fi
        
        if [ -f "test_processed_local_checkpoint.json" ]; then
            python3 optimized_batch_processor_local.py \
                --input test_samples.json \
                --output test_processed_local.json \
                --server-url "$VLLM_SERVER_URL" \
                --checkpoint-interval 25
        else
            echo "未找到测试集检查点文件"
        fi
        ;;
    7)
        echo "启动VLLM服务器..."
        start_vllm_server
        if [ $? -eq 0 ]; then
            echo "VLLM服务器已启动并运行在 $VLLM_SERVER_URL"
            echo "日志文件: vllm_server.log"
            echo "要停止服务器，请运行此脚本并选择选项8"
        fi
        ;;
    8)
        echo "停止VLLM服务器..."
        stop_vllm_server
        ;;
    9)
        check_vllm_server
        if [ -f "vllm_server.log" ]; then
            echo ""
            echo "=== 最近的服务器日志 ==="
            tail -n 10 vllm_server.log
        fi
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo "操作完成！" 
