#!/bin/bash

# Set exit on error
set -e

# Define task list (from TASK_CONFIG in evaluator.py)
TASKS=(
    "rectified_scaling_law"
    "domain_mixture_scaling_law"
    "data_constrained_scaling_law"
    "moe_scaling_law"
    "vocab_scaling_law"
)

# Define number of runs
RUNS=(0 1 2)

# Maximum parallel tasks (can be adjusted based on system resources)
MAX_PARALLEL_TASKS=5

# 颜色输出函数
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# 检查openevolve-run命令是否可用
check_openevolve() {
    if ! command -v openevolve-run &> /dev/null; then
        print_error "openevolve-run 命令未找到，请确保已正确安装"
        exit 1
    fi
}

# 等待后台任务完成
wait_for_jobs() {
    local max_jobs=$1
    while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
        sleep 1
    done
}

# 处理单个任务的函数
process_task() {
    local task=$1
    local task_id=$2
    
    print_info "[任务 $task_id] 开始处理任务: $task"
    
    # 检查任务目录是否存在
    if [ ! -d "$task" ]; then
        print_error "[任务 $task_id] 任务目录 $task 不存在，跳过"
        return 1
    fi
    
    # 检查必要的文件是否存在
    if [ ! -f "$task/config.yaml" ]; then
        print_error "[任务 $task_id] 配置文件 $task/config.yaml 不存在，跳过"
        return 1
    fi
    
    if [ ! -f "$task/init_program.py" ]; then
        print_error "[任务 $task_id] 初始化程序 $task/init_program.py 不存在，跳过"
        return 1
    fi
    
    # 进入任务目录
    cd "$task"
    
    # 运行3次进化算法
    for i in "${RUNS[@]}"; do
        print_info "[任务 $task_id] 运行第 $((i+1)) 次进化算法 (i=$i)"
        
        # 创建输出目录
        mkdir -p "outputs/evosld_$i"
        
        # 检查最佳程序是否已存在
        best_program_path="outputs/evosld_$i/best/best_program.py"
        if [ -f "$best_program_path" ]; then
            print_info "[任务 $task_id] 最佳程序已存在: $best_program_path，跳过进化算法运行"
        else
            # 运行openevolve-run
            if openevolve-run --config config.yaml init_program.py evaluator.py --output "outputs/evosld_$i"; then
                print_success "[任务 $task_id] 任务 $task 第 $((i+1)) 次运行完成"
            else
                print_error "[任务 $task_id] 任务 $task 第 $((i+1)) 次运行失败"
            fi
        fi
    done
    
    # 为每次运行进行评估
    print_info "[任务 $task_id] 开始评估任务: $task"
    
    for i in "${RUNS[@]}"; do
        print_info "[任务 $task_id] 评估第 $((i+1)) 次运行结果 (i=$i)"
        
        # 检查最佳程序是否存在（重新定义路径变量）
        best_program_path="outputs/evosld_$i/best/best_program.py"
        if [ ! -f "$best_program_path" ]; then
            print_warning "[任务 $task_id] 最佳程序 $best_program_path 不存在，跳过评估"
            continue
        fi
        
        # 运行评估并保存到日志文件（追加模式）
        log_file="outputs/evosld_$i/evaluation_log.txt"
        print_info "[任务 $task_id] 运行评估: python ../evaluator.py $task $best_program_path"
        
        if python ../evaluator.py "$task" "$best_program_path" >> "$log_file" 2>&1; then
            print_success "[任务 $task_id] 任务 $task 第 $((i+1)) 次评估完成，日志保存到 $log_file"
            
            # 显示评估结果摘要（追加模式）
            echo "=== 评估结果摘要 ===" >> "$log_file"
            echo "Task: $task" >> "$log_file"
            echo "运行次数: $((i+1))" >> "$log_file"
            echo "评估时间: $(date)" >> "$log_file"
            echo "==================" >> "$log_file"
        else
            print_error "[任务 $task_id] 任务 $task 第 $((i+1)) 次评估失败"
        fi
    done
    
    # 返回上级目录
    cd ..
    
    print_success "[任务 $task_id] 任务 $task 的所有处理完成"
}

# 主函数
main() {
    print_info "开始运行SLD任务自动化脚本（并行模式）"
    print_info "任务列表: ${TASKS[*]}"
    print_info "每个任务运行次数: ${RUNS[*]}"
    print_info "最大并行任务数: $MAX_PARALLEL_TASKS"
    
    # 检查openevolve-run
    check_openevolve
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 存储后台任务ID
    declare -a task_pids
    local task_id=1
    
    # 为每个任务启动后台进程
    for task in "${TASKS[@]}"; do
        # 等待可用槽位
        wait_for_jobs $MAX_PARALLEL_TASKS
        
        # 启动后台任务
        process_task "$task" "$task_id" &
        task_pids+=($!)
        
        print_info "启动任务 $task_id ($task) 在后台，PID: $!"
        ((task_id++))
    done
    
    # 等待所有后台任务完成
    print_info "等待所有任务完成..."
    for pid in "${task_pids[@]}"; do
        wait $pid
        print_info "任务 PID $pid 已完成"
    done
    
    # 计算总运行时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_success "所有任务完成！"
    print_info "总运行时间: ${DURATION} 秒 ($(($DURATION / 60)) 分钟)"
    
    # 生成总结报告
    generate_summary_report
}

# 生成总结报告
generate_summary_report() {
    print_info "生成总结报告..."
    
    report_file="run_summary_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "SLD任务运行总结报告（并行模式）"
        echo "=============================="
        echo "运行时间: $(date)"
        echo "任务列表: ${TASKS[*]}"
        echo "每个任务运行次数: ${RUNS[*]}"
        echo "最大并行任务数: $MAX_PARALLEL_TASKS"
        echo ""
        
        for task in "${TASKS[@]}"; do
            echo "Task: $task"
            echo "----------------"
            
            if [ -d "$task" ]; then
                for i in "${RUNS[@]}"; do
                    log_file="$task/outputs/evosld_$i/evaluation_log.txt"
                    if [ -f "$log_file" ]; then
                        echo "  运行 $((i+1)): 评估完成，日志文件: $log_file"
                    else
                        echo "  运行 $((i+1)): 评估未完成或失败"
                    fi
                done
            else
                echo "  任务目录不存在"
            fi
            echo ""
        done
    } > "$report_file"
    
    print_success "总结报告已保存到: $report_file"
}

# 运行主函数
main "$@"