#!/usr/bin/env python3
"""
优化的批量处理脚本 - 本地VLLM版本
针对7400条样本数据进行处理，支持中断恢复、增量保存等功能
使用本地VLLM部署的模型进行推理
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
from smart_rule_matcher_local import SmartMedicalRuleMatcherLocal, default_mimic_reports_path


STAGE3_DIR = Path(__file__).resolve().parent
CODE_ROOT = STAGE3_DIR.parent

class OptimizedBatchProcessorLocal:
    def __init__(
        self,
        local_server_url: str,
        cxr_rules_path: str,
        mimic_cxr_path: str,
        model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        disease_mapping_path: str = "stage3/disease_in_MIMIC_mapto_WHO.csv",
        dry_run: bool = False,
        log_dir: str = "outputs/logs",
    ):
        """
        初始化优化的批量处理器（本地VLLM版本）
        
        Args:
            local_server_url: 本地VLLM服务器URL
            cxr_rules_path: CXR规则文件路径
            mimic_cxr_path: MIMIC-CXR数据路径
            model: 本地OpenAI兼容服务中的模型名称
            disease_mapping_path: VQA疾病到WHO疾病的映射CSV路径
            dry_run: 不调用本地LLM，生成确定性占位结果
            log_dir: 日志目录
        """
        self.matcher = SmartMedicalRuleMatcherLocal(
            local_server_url,
            model=model,
            disease_mapping_path=disease_mapping_path,
            dry_run=dry_run,
            skip_server_check=dry_run,
        )
        self.matcher.load_cxr_rules(cxr_rules_path)
        self.mimic_cxr_path = mimic_cxr_path
        self.dry_run = dry_run
        
        # 设置日志
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = str(Path(log_dir) / f"batch_process_local_{int(time.time())}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"本地VLLM批量处理器日志文件: {log_file}")
        
    def load_samples(self, input_file: str) -> list:
        """加载样本数据"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.logger.info(f"成功加载 {len(data)} 个样本")
        return data
    
    def create_checkpoint(self, processed_data: list, checkpoint_file: str):
        """创建检查点文件"""
        Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    def load_checkpoint(self, checkpoint_file: str) -> list:
        """加载检查点文件"""
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.info(f"从检查点恢复 {len(data)} 个已处理样本")
                return data
            except Exception as e:
                self.logger.warning(f"加载检查点失败: {e}")
        return []
    
    def process_samples(self, input_file: str, output_file: str, 
                       checkpoint_interval: int = 100, start_idx: int = 0,
                       max_samples: int = None):
        """
        批量处理样本
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            checkpoint_interval: 检查点保存间隔
            start_idx: 开始处理的索引（用于手动指定恢复点）
        """
        # 设置检查点文件
        checkpoint_file = output_file.replace('.json', '_checkpoint.json')
        
        # 加载原始样本
        all_samples = self.load_samples(input_file)
        if max_samples is not None:
            all_samples = all_samples[:max_samples]
        total_samples = len(all_samples)
        
        # 尝试从检查点恢复
        processed_data = self.load_checkpoint(checkpoint_file)
        processed_count = len(processed_data)
        
        # 确定开始位置
        start_idx = max(start_idx, processed_count)
        
        if start_idx > 0:
            self.logger.info(f"从索引 {start_idx} 开始处理（已处理 {processed_count} 个样本）")
        
        # 处理样本
        successful_count = 0
        failed_count = 0
        
        # 创建进度条
        pbar = tqdm(
            range(start_idx, total_samples),
            desc=f"处理 {Path(input_file).stem}",
            initial=start_idx,
            total=total_samples
        )
        
        try:
            for idx in pbar:
                sample = all_samples[idx]
                
                # 更新进度条描述
                pbar.set_description(f"处理样本 {idx+1}/{total_samples} (成功:{successful_count}, 失败:{failed_count})")
                
                try:
                    # 分析样本
                    processed_sample = self.matcher.analyze_sample(sample, self.mimic_cxr_path)
                    processed_data.append(processed_sample)
                    successful_count += 1
                    
                    # 本地推理通常更快，可以适当减少延迟
                    time.sleep(0.1)
                    
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"处理样本 {idx} 失败: {e}")
                    
                    # 保存失败的样本（标记错误）
                    failed_sample = sample.copy()
                    failed_sample['global_who_rules'] = []
                    failed_sample['personal_who_rules'] = []
                    failed_sample['medical_report_found'] = False
                    failed_sample['processing_error'] = str(e)
                    failed_sample['processing_index'] = idx
                    
                    processed_data.append(failed_sample)
                
                # 定期保存检查点
                if (idx + 1) % checkpoint_interval == 0:
                    self.create_checkpoint(processed_data, checkpoint_file)
                    self.logger.info(f"已保存检查点：{idx + 1}/{total_samples} 个样本")
                
                # 如果连续失败太多次，暂停一会（可能是本地服务器问题）
                if failed_count > 0 and failed_count % 20 == 0:
                    self.logger.warning(f"已失败 {failed_count} 次，暂停10秒检查本地服务器...")
                    time.sleep(10)
        
        except KeyboardInterrupt:
            self.logger.info(f"用户中断处理，已处理到索引 {idx}")
            self.create_checkpoint(processed_data, checkpoint_file)
            pbar.close()
            return processed_data, successful_count, failed_count
        
        pbar.close()
        
        # 保存最终结果
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        # 删除检查点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            
        self.logger.info(f"处理完成！成功: {successful_count}, 失败: {failed_count}")
        self.logger.info(f"结果已保存到: {output_file}")
        
        return processed_data, successful_count, failed_count
    
    def validate_output(self, output_file: str):
        """验证输出文件的完整性"""
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查基本统计
            total_count = len(data)
            with_global_rules = sum(1 for item in data if item.get('global_who_rules'))
            with_personal_rules = sum(1 for item in data if item.get('personal_who_rules'))
            with_medical_reports = sum(1 for item in data if item.get('medical_report_found', False))
            with_errors = sum(1 for item in data if 'processing_error' in item)
            
            self.logger.info(f"输出文件验证结果:")
            self.logger.info(f"  总样本数: {total_count}")
            self.logger.info(f"  有全局规则: {with_global_rules}")
            self.logger.info(f"  有个人规则: {with_personal_rules}")
            self.logger.info(f"  找到医疗报告: {with_medical_reports}")
            self.logger.info(f"  处理错误: {with_errors}")
            
            # 检查数据完整性
            required_fields = ['global_who_rules', 'personal_who_rules']
            missing_fields = []
            for i, item in enumerate(data):
                for field in required_fields:
                    if field not in item:
                        missing_fields.append(f"样本{i}缺少字段{field}")
            
            if missing_fields:
                self.logger.warning(f"发现 {len(missing_fields)} 个字段缺失问题")
                for msg in missing_fields[:10]:  # 只显示前10个
                    self.logger.warning(f"  {msg}")
            else:
                self.logger.info("所有样本都包含必需字段")
            
            return True
        except Exception as e:
            self.logger.error(f"验证输出文件失败: {e}")
            return False
    
    def analyze_disease_distribution(self, output_file: str):
        """分析疾病分布统计"""
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 统计VQA答案中的疾病
            vqa_diseases = {}
            for item in data:
                answers = item.get('answer', [])
                if isinstance(answers, str):
                    answers = [answers]
                for answer in answers:
                    disease = answer.lower().strip()
                    vqa_diseases[disease] = vqa_diseases.get(disease, 0) + 1
            
            # 统计匹配到的WHO规则
            who_rule_diseases = {}
            for item in data:
                global_rules = item.get('global_who_rules', [])
                for rule in global_rules:
                    disease = rule.get('disease', '').lower().strip()
                    if disease:
                        who_rule_diseases[disease] = who_rule_diseases.get(disease, 0) + 1
            
            # 输出统计结果
            self.logger.info(f"疾病分布分析:")
            self.logger.info(f"  VQA答案中的疾病种类: {len(vqa_diseases)}")
            self.logger.info(f"  匹配到WHO规则的疾病种类: {len(who_rule_diseases)}")
            
            # 显示最常见的疾病
            sorted_vqa = sorted(vqa_diseases.items(), key=lambda x: x[1], reverse=True)
            self.logger.info(f"  最常见的VQA疾病:")
            for disease, count in sorted_vqa[:10]:
                self.logger.info(f"    {disease}: {count}")
            
            return vqa_diseases, who_rule_diseases
            
        except Exception as e:
            self.logger.error(f"疾病分布分析失败: {e}")
            return {}, {}

def main():
    parser = argparse.ArgumentParser(description='优化的MIMIC-CXR-VQA样本批量处理（本地VLLM版本）')
    parser.add_argument('--input', required=True, help='输入JSON文件路径')
    parser.add_argument('--output', required=True, help='输出JSON文件路径')
    parser.add_argument('--server-url', default='http://localhost:8000', 
                       help='本地VLLM服务器URL')
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
                       help='本地OpenAI兼容服务中的模型名称')
    parser.add_argument('--cxr-rules', default='stage3/cxr_grade_er_conversion_final_20250712_200006.json',
                       help='CXR规则文件路径')
    parser.add_argument('--disease-mapping', default='stage3/disease_in_MIMIC_mapto_WHO.csv',
                       help='VQA疾病到WHO疾病的映射CSV路径')
    parser.add_argument('--mimic-cxr-path', 
                       default=default_mimic_reports_path(),
                       help='MIMIC-CXR数据路径')
    parser.add_argument('--start-idx', type=int, default=0, help='开始处理的索引')
    parser.add_argument('--checkpoint-interval', type=int, default=100, help='检查点保存间隔')
    parser.add_argument('--max-samples', type=int, default=None, help='最多处理的样本数')
    parser.add_argument('--dry-run', action='store_true', help='不调用本地LLM，生成确定性占位结果')
    parser.add_argument('--log-dir', default='outputs/logs', help='日志目录')
    parser.add_argument('--validate', action='store_true', help='只验证输出文件')
    parser.add_argument('--analyze', action='store_true', help='分析疾病分布')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = OptimizedBatchProcessorLocal(
        args.server_url,
        args.cxr_rules,
        args.mimic_cxr_path,
        model=args.model,
        disease_mapping_path=args.disease_mapping,
        dry_run=args.dry_run,
        log_dir=args.log_dir,
    )
    
    if args.validate:
        # 只验证输出文件
        processor.validate_output(args.output)
        if args.analyze:
            processor.analyze_disease_distribution(args.output)
    else:
        # 处理样本
        data, success, failed = processor.process_samples(
            args.input,
            args.output,
            args.checkpoint_interval,
            args.start_idx,
            args.max_samples,
        )
        
        # 验证结果
        processor.validate_output(args.output)
        
        # 分析疾病分布
        processor.analyze_disease_distribution(args.output)

if __name__ == "__main__":
    main() 
