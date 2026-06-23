#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Medical Rule Matcher - Local VLLM Version
两阶段WHO指南规则匹配：全局规则提取 + 基于报告语义的局部黄金规则匹配
使用本地VLLM部署的模型进行推理
"""

import json
import csv
import os
import requests
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
import re
from pathlib import Path


STAGE3_DIR = Path(__file__).resolve().parent
CODE_ROOT = STAGE3_DIR.parent
DEFAULT_DISEASE_MAPPING = STAGE3_DIR / "disease_in_MIMIC_mapto_WHO.csv"


def default_mimic_reports_path() -> str:
    """Return the canonical MIMIC-CXR-JPG report path without hard-coded host paths."""
    return str(Path(os.getenv("MIMIC_CXR_JPG_ROOT", "data/mimic-cxr-jpg")) / "files")


def resolve_stage3_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    root_candidate = CODE_ROOT / candidate
    if root_candidate.exists():
        return root_candidate
    stage3_candidate = STAGE3_DIR / candidate
    if stage3_candidate.exists():
        return stage3_candidate
    return candidate


class SmartMedicalRuleMatcherLocal:
    def __init__(self, local_server_url: str = "http://localhost:8000", 
                 model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                 disease_mapping_path: Optional[str] = None,
                 dry_run: bool = False,
                 skip_server_check: bool = False):
        """
        初始化智能医疗规则匹配器（本地VLLM版本）
        
        Args:
            local_server_url: 本地VLLM服务器URL
            model: 使用的模型名称（用于请求中的模型字段）
            disease_mapping_path: MIMIC疾病到WHO疾病映射CSV路径
            dry_run: 不调用本地LLM，生成确定性占位结果
            skip_server_check: 跳过本地VLLM健康检查
        """
        self.local_server_url = local_server_url.rstrip('/')
        self.model = model
        self.dry_run = dry_run
        self.cxr_rules = {}
        self.disease_mapping = {}  # 添加疾病映射
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 检查本地服务器连接
        if not self.dry_run and not skip_server_check:
            self._check_server_connection()
        
        # 加载疾病映射文件
        self._load_disease_mapping(disease_mapping_path)
    
    def _check_server_connection(self):
        """检查本地服务器连接"""
        try:
            response = requests.get(f"{self.local_server_url}/health", timeout=5)
            if response.status_code == 200:
                self.logger.info(f"成功连接到本地VLLM服务器: {self.local_server_url}")
            else:
                self.logger.warning(f"本地服务器响应异常，状态码: {response.status_code}")
        except Exception as e:
            self.logger.error(f"无法连接到本地VLLM服务器 {self.local_server_url}: {e}")
            self.logger.error("请确保VLLM服务器已启动")
    
    def load_cxr_rules(self, cxr_rules_path: str):
        """加载CXR规则数据"""
        try:
            rules_path = resolve_stage3_path(cxr_rules_path)
            with open(rules_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # 提取实际的规则数据
            if 'diseases' in raw_data:
                # 重构规则为扁平结构
                self.cxr_rules = {}
                for disease_name, disease_data in raw_data['diseases'].items():
                    if 'formalized_cxr_rules' in disease_data:
                        for rule in disease_data['formalized_cxr_rules']:
                            rule_id = rule.get('rule_id', f'RULE_{disease_name.upper().replace(" ", "_")}')
                            self.cxr_rules[rule_id] = {
                                'disease': disease_name,
                                'original_statement': rule.get('original_statement', ''),
                                'rule_id': rule_id,
                                'source_chapter': rule.get('source_chapter', ''),
                                'cxr_field_type': rule.get('cxr_field_type', ''),
                                'subfield_type': rule.get('subfield_type', ''),
                                'atomic_propositions': rule.get('atomic_propositions', [])
                            }
            else:
                # 如果是旧格式，直接使用
                self.cxr_rules = raw_data
            
            self.logger.info(f"成功加载CXR规则: {len(self.cxr_rules)}个规则")
        except Exception as e:
            self.logger.error(f"加载CXR规则失败: {e}")
            raise
    
    def _load_disease_mapping(self, mapping_file: Optional[str] = None):
        """加载MIMIC疾病到WHO疾病的映射文件"""
        import csv
        mapping_path = Path(mapping_file) if mapping_file else DEFAULT_DISEASE_MAPPING
        if not mapping_path.is_absolute() and not mapping_path.exists():
            root_candidate = CODE_ROOT / mapping_path
            if root_candidate.exists():
                mapping_path = root_candidate
        if not mapping_path.is_absolute() and not mapping_path.exists():
            stage3_candidate = STAGE3_DIR / mapping_path
            if stage3_candidate.exists():
                mapping_path = stage3_candidate
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    abnormality = row['Abnormality'].lower().strip()
                    path = row['Hierarchical_Path']
                    
                    if abnormality not in self.disease_mapping:
                        self.disease_mapping[abnormality] = []
                    
                    # 提取最终的疾病名称
                    if '->' in path:
                        final_disease = path.split('->')[-1].strip()
                        # 清理标记
                        final_disease = final_disease.replace('###', '').replace('##', '').replace('#', '').strip()
                    else:
                        final_disease = path.replace('#', '').strip()
                    
                    self.disease_mapping[abnormality].append(final_disease)
            
            self.logger.info(f"成功加载疾病映射: {len(self.disease_mapping)} 个VQA疾病")
            
        except Exception as e:
            self.logger.warning(f"加载疾病映射文件失败: {e}")
            self.disease_mapping = {}
    
    def extract_global_who_rules(self, vqa_answers: List[str]) -> List[Dict]:
        """
        第一阶段：根据VQA答案提取全局WHO规则
        使用映射文件正确匹配疾病规则
        """
        if not vqa_answers:
            return []
        
        global_rules = []
        
        # 从答案中提取疾病名称
        for answer in vqa_answers:
            cleaned_answer = answer.lower().strip()
            
            # 查找映射关系
            mapped_diseases = []
            if cleaned_answer in self.disease_mapping:
                mapped_diseases = self.disease_mapping[cleaned_answer]
                self.logger.debug(f"找到映射: {cleaned_answer} → {mapped_diseases}")
            else:
                # 如果没有找到精确匹配，尝试部分匹配
                for mapped_vqa_disease, who_diseases in self.disease_mapping.items():
                    if (cleaned_answer in mapped_vqa_disease or 
                        mapped_vqa_disease in cleaned_answer or
                        self._are_diseases_related(cleaned_answer, mapped_vqa_disease)):
                        mapped_diseases.extend(who_diseases)
                        self.logger.debug(f"找到部分匹配: {cleaned_answer} ≈ {mapped_vqa_disease} → {who_diseases}")
                        break
            
            # 在CXR规则中查找对应的规则
            if mapped_diseases:
                for who_disease in mapped_diseases:
                    # 在CXR规则中查找匹配的疾病名称
                    for rule_id, rule_data in self.cxr_rules.items():
                        cxr_disease_name = rule_data.get('disease', '').lower()
                        who_disease_lower = who_disease.lower()
                        
                        # 检查匹配
                        if (who_disease_lower in cxr_disease_name or 
                            cxr_disease_name in who_disease_lower or
                            self._are_diseases_related(who_disease_lower, cxr_disease_name)):
                            
                            global_rules.append({
                                'rule_id': rule_id,
                                'disease': rule_data.get('disease', ''),
                                'original_statement': rule_data.get('original_statement', ''),
                                'source': 'CXR_RULE',
                                'matched_answer': cleaned_answer,
                                'mapped_who_disease': who_disease
                            })
                            self.logger.debug(f"匹配规则: {cleaned_answer} → {who_disease} → {rule_data.get('disease', '')}")
            else:
                # 如果映射文件中没有找到，回退到原来的直接匹配方式
                self.logger.warning(f"映射文件中未找到 '{cleaned_answer}'，使用直接匹配")
                for rule_id, rule_data in self.cxr_rules.items():
                    disease_name = rule_data.get('disease', '').lower()
                    
                    if (disease_name in cleaned_answer or 
                        cleaned_answer in disease_name or
                        self._are_diseases_related(cleaned_answer, disease_name)):
                        
                        global_rules.append({
                            'rule_id': rule_id,
                            'disease': rule_data.get('disease', ''),
                            'original_statement': rule_data.get('original_statement', ''),
                            'source': 'CXR_RULE',
                            'matched_answer': cleaned_answer,
                            'mapped_who_disease': 'Direct_Match'
                        })
                        break
        
        # 去重（基于rule_id）
        seen_rules = set()
        unique_rules = []
        for rule in global_rules:
            if rule['rule_id'] not in seen_rules:
                seen_rules.add(rule['rule_id'])
                unique_rules.append(rule)
        
        self.logger.info(f"提取到 {len(unique_rules)} 条全局WHO规则")
        return unique_rules
    
    def _are_diseases_related(self, disease1: str, disease2: str) -> bool:
        """判断两个疾病名称是否相关"""
        # 移除常见的医学后缀/前缀进行比较
        def normalize_disease(name):
            # 移除常见词汇
            words_to_remove = ['disease', 'syndrome', 'disorder', 'condition', 'acute', 'chronic']
            words = name.split()
            filtered_words = [word for word in words if word not in words_to_remove]
            return ' '.join(filtered_words)
        
        norm1 = normalize_disease(disease1)
        norm2 = normalize_disease(disease2)
        
        # 检查是否有共同的关键词
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        return len(words1.intersection(words2)) > 0
    
    def consolidate_with_medical_report(self, global_rules: List[Dict], 
                                      medical_report: str, 
                                      max_retries: int = 3) -> List[Dict]:
        """
        第二阶段：使用本地VLLM将全局规则与医疗报告进行语义整合
        添加重试机制处理连接问题
        """
        if not global_rules or not medical_report.strip():
            self.logger.warning("全局规则或医疗报告为空，跳过整合")
            return []

        if self.dry_run:
            return self._make_dry_run_personal_rules(global_rules, medical_report)
        
        # 如果规则太多，分批处理
        batch_size = 10  # 每批最多处理10条规则
        all_personal_rules = []
        
        for i in range(0, len(global_rules), batch_size):
            batch_rules = global_rules[i:i+batch_size]
            self.logger.info(f"处理规则批次 {i//batch_size + 1}/{(len(global_rules)-1)//batch_size + 1}")
            
            for retry in range(max_retries):
                try:
                    batch_personal_rules = self._process_rule_batch_local(batch_rules, medical_report)
                    all_personal_rules.extend(batch_personal_rules)
                    break
                except Exception as e:
                    self.logger.warning(f"批次处理失败 (尝试 {retry+1}/{max_retries}): {e}")
                    if retry == max_retries - 1:
                        self.logger.error(f"批次处理最终失败，跳过此批次: {e}")
                        # 添加占位符表示失败
                        for rule in batch_rules:
                            all_personal_rules.append({
                                'rule_id': rule['rule_id'],
                                'original_statement': rule['original_statement'],
                                'semantic_evidence': "处理失败：本地服务器错误",
                                'confidence_rating': 0.0,
                                'detailed_reasoning': "由于本地服务器问题无法处理此规则",
                                'error': str(e)
                            })
                    else:
                        time.sleep(2 ** retry)  # 指数退避
        
        return all_personal_rules

    def _make_dry_run_personal_rules(self, rules: List[Dict], medical_report: str) -> List[Dict]:
        """Generate deterministic Stage 3-shaped output without a local LLM server."""
        report_lower = medical_report.lower()
        report_excerpt = re.sub(r"\s+", " ", medical_report).strip()[:240]
        personal_rules = []
        for rule in rules:
            statement = rule.get('original_statement', '')
            disease = rule.get('disease', '')
            keywords = [
                token
                for token in re.findall(r"[a-zA-Z]{4,}", f"{disease} {statement}".lower())
                if token not in {"with", "from", "that", "this", "have", "show", "shows"}
            ]
            overlap = sorted({token for token in keywords if token in report_lower})
            confidence = 0.75 if overlap else 0.25
            personal_rules.append({
                'rule_id': rule.get('rule_id', ''),
                'original_statement': statement,
                'semantic_evidence': f"Dry-run report excerpt: {report_excerpt}",
                'confidence_rating': confidence,
                'detailed_reasoning': (
                    "Dry-run lexical overlap used for pipeline validation. "
                    f"Matched terms: {', '.join(overlap[:8]) if overlap else 'none'}."
                ),
                'dry_run': True
            })
        return personal_rules
    
    def _process_rule_batch_local(self, rules: List[Dict], medical_report: str) -> List[Dict]:
        """使用本地VLLM服务器处理一批规则"""
        rules_text = ""
        for rule in rules:
            rules_text += f"Rule ID: {rule['rule_id']}\n"
            rules_text += f"Disease: {rule['disease']}\n"
            rules_text += f"Statement: {rule['original_statement']}\n\n"
        
        prompt = f"""You are a senior medical expert specializing in radiology and clinical diagnostics. Your task is to perform semantic analysis to determine which WHO medical guidelines were actually applied or referenced in a given medical report.

**MEDICAL REPORT:**
{medical_report}

**WHO GUIDELINES TO ANALYZE:**
{rules_text}

**CRITICAL INSTRUCTIONS:**
1. Analyze each WHO guideline to determine if the medical report semantically aligns with or references the guideline
2. For EACH guideline, you MUST provide a response entry, even if confidence is 0
3. Your response MUST be a valid, complete JSON array
4. DO NOT add any explanatory text before or after the JSON
5. DO NOT use markdown formatting or code blocks
6. Ensure the JSON is properly formatted with correct brackets and commas

**RESPONSE FORMAT (JSON ARRAY ONLY):**
[
  {{
    "rule_id": "exact_rule_id_here",
    "original_statement": "exact_rule_statement_here",
    "semantic_evidence": "specific_evidence_from_report",
    "confidence_rating": 0.0_to_1.0,
    "detailed_reasoning": "brief_reasoning_explanation"
  }}
]

**ANALYSIS CRITERIA:**
- Explicit References: Direct mentions of diagnostic criteria or findings
- Semantic Alignment: Clinical observations that correspond to guideline recommendations
- Confidence Scale: 0.0 (no correlation) to 1.0 (perfect match)

Generate the JSON response now:"""

        # 构造请求数据
        request_data = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.0,  # 降低温度以获得更一致的响应
            'stream': False,
            'max_tokens': 8192,  # 设置合理的最大token数
            'top_p': 0.9,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
        
        # 发送请求到本地VLLM服务器
        try:
            response = requests.post(
                f'{self.local_server_url}/v1/chat/completions',
                headers={'Content-Type': 'application/json'},
                json=request_data,
                timeout=120  # 2分钟超时
            )
            
            if response.status_code != 200:
                raise Exception(f"本地服务器请求失败: {response.status_code}, {response.text}")
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
        except requests.exceptions.Timeout:
            raise Exception("本地服务器请求超时")
        except requests.exceptions.ConnectionError:
            raise Exception("无法连接到本地服务器")
        except Exception as e:
            raise Exception(f"本地服务器请求失败: {e}")
        
        try:
            # 清理响应内容，提取JSON部分
            content = content.strip()
            
            # 移除可能的markdown代码块标记
            content = content.replace('```json', '').replace('```', '')
            
            # 移除可能的思考标签和其他额外内容
            if '<think>' in content:
                think_start = content.find('<think>')
                think_end = content.find('</think>')
                if think_start != -1 and think_end != -1:
                    content = content[:think_start] + content[think_end + 8:]
            
            # 清理其他标签
            content = content.replace('<think>', '').replace('</think>', '')
            
            # 寻找JSON数组的开始和结束
            start_idx = content.find('[')
            if start_idx == -1:
                self.logger.error(f"未找到JSON数组开始标记: {content[:200]}...")
                return []
            
            # 寻找匹配的结束标记
            bracket_count = 0
            end_idx = -1
            for i in range(start_idx, len(content)):
                if content[i] == '[':
                    bracket_count += 1
                elif content[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx == -1:
                self.logger.warning("JSON数组未正确闭合，尝试修复...")
                # 尝试修复截断的JSON
                json_part = content[start_idx:]
                
                # 如果找到最后一个完整的对象，在它后面加上闭合标记
                last_complete_obj = json_part.rfind('}')
                if last_complete_obj != -1:
                    # 检查是否有逗号
                    next_char_idx = last_complete_obj + 1
                    while next_char_idx < len(json_part) and json_part[next_char_idx] in [' ', '\n', '\t']:
                        next_char_idx += 1
                    
                    if next_char_idx < len(json_part) and json_part[next_char_idx] == ',':
                        # 移除末尾的逗号并添加闭合标记
                        json_part = json_part[:next_char_idx] + '\n]'
                    else:
                        # 直接添加闭合标记
                        json_part = json_part[:last_complete_obj + 1] + '\n]'
                else:
                    self.logger.error("无法修复截断的JSON")
                    return []
            else:
                json_part = content[start_idx:end_idx]
            
            self.logger.debug(f"提取的JSON: {json_part[:500]}...")
            
            # 尝试解析JSON
            try:
                parsed_result = json.loads(json_part)
                if isinstance(parsed_result, list):
                    self.logger.info(f"成功解析JSON，包含 {len(parsed_result)} 个条目")
                    return parsed_result
                else:
                    self.logger.error(f"解析结果不是数组: {type(parsed_result)}")
                    return []
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON解析失败: {e}")
                self.logger.error(f"提取的JSON: {json_part}")
                
                # 尝试更积极的修复
                try:
                    # 移除可能的注释和额外内容
                    lines = json_part.split('\n')
                    clean_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('//') and not line.startswith('#'):
                            clean_lines.append(line)
                    
                    clean_json = '\n'.join(clean_lines)
                    parsed_result = json.loads(clean_json)
                    
                    if isinstance(parsed_result, list):
                        self.logger.info(f"修复后成功解析JSON，包含 {len(parsed_result)} 个条目")
                        return parsed_result
                    else:
                        self.logger.error(f"修复后解析结果不是数组: {type(parsed_result)}")
                        return []
                        
                except json.JSONDecodeError as e2:
                    self.logger.error(f"修复后仍然解析失败: {e2}")
                    return []
             
        except Exception as e:
            self.logger.error(f"处理响应时发生错误: {e}")
            return []
    
    def get_medical_report_by_ids(
        self,
        subject_id: str,
        study_id: str,
        mimic_cxr_path: Optional[str] = None
    ) -> str:
        """根据subject_id和study_id获取医疗报告"""
        try:
            mimic_cxr_path = mimic_cxr_path or default_mimic_reports_path()
            subject_id = str(subject_id)
            study_id = str(study_id)
            # 构建路径 - 修正路径结构，添加s前缀
            subject_dir = f"p{subject_id[:2]}/p{subject_id}"
            report_file = f"s{study_id}.txt"
            
            report_path = Path(mimic_cxr_path) / subject_dir / report_file
            
            if not report_path.exists():
                self.logger.warning(f"报告文件不存在: {report_path}")
                return ""
            
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取FINDINGS和IMPRESSION部分
            findings_match = re.search(r'FINDINGS:\s*(.*?)(?=IMPRESSION:|$)', content, re.DOTALL | re.IGNORECASE)
            impression_match = re.search(r'IMPRESSION:\s*(.*?)(?=$)', content, re.DOTALL | re.IGNORECASE)
            
            report_parts = []
            if findings_match:
                report_parts.append(f"FINDINGS: {findings_match.group(1).strip()}")
            if impression_match:
                report_parts.append(f"IMPRESSION: {impression_match.group(1).strip()}")
            
            return '\n\n'.join(report_parts) if report_parts else content
            
        except Exception as e:
            self.logger.error(f"读取医疗报告失败 {subject_id}/{study_id}: {e}")
            return ""
    
    def analyze_sample(
        self,
        sample: Dict,
        mimic_cxr_path: Optional[str] = None
    ) -> Dict:
        """
        分析单个样本，返回包含原始数据和分析结果的完整字典
        """
        try:
            # 提取VQA答案
            vqa_answers = sample.get('answer', [])
            if isinstance(vqa_answers, str):
                vqa_answers = [vqa_answers]
            
            # 获取医疗报告
            subject_id = sample.get('subject_id', '')
            study_id = sample.get('study_id', '')
            medical_report = sample.get('report_text') or sample.get('medical_report') or ""
            if not medical_report:
                medical_report = self.get_medical_report_by_ids(subject_id, study_id, mimic_cxr_path)
            
            # 第一阶段：提取全局WHO规则
            global_rules = self.extract_global_who_rules(vqa_answers)
            
            # 第二阶段：语义整合
            personal_rules = []
            if medical_report.strip() and global_rules:
                personal_rules = self.consolidate_with_medical_report(global_rules, medical_report)
            
            # 创建结果字典，保留原始数据并添加新字段
            result = sample.copy()
            result['global_who_rules'] = global_rules
            result['personal_who_rules'] = personal_rules
            result['medical_report_found'] = bool(medical_report.strip())
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析样本失败: {e}")
            # 即使出错也返回原始数据，但标记错误
            result = sample.copy()
            result['global_who_rules'] = []
            result['personal_who_rules'] = []
            result['medical_report_found'] = False
            result['analysis_error'] = str(e)
            return result

# 使用示例
if __name__ == "__main__":
    # 初始化匹配器（本地版本）
    matcher = SmartMedicalRuleMatcherLocal("http://localhost:8000")
    
    # 加载数据
    matcher.load_cxr_rules("stage3/cxr_grade_er_conversion_final_20250712_200006.json")
    
    print("智能医疗规则匹配器（本地VLLM版本）初始化完成！") 
