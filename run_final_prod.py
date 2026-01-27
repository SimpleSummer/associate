import pandas as pd
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import torch
import re
import os
import sys

# ================= 1. 全局配置区域 =================

# 模型配置：多语言对齐模型 (CPU 友好，准确度高)
MODEL_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_MODEL_DIR = "./multilingual-minilm-local" 

# 输入输出文件配置 (请根据实际文件名修改)
# 建议使用 CSV 格式以支持 100万+ 行数据
TABLE_FILE_IN = "tables_test.csv"        # 输入：表清单
TABLE_FILE_OUT = "tables_result.csv"     # 输出：表清单结果

COLUMN_FILE_IN = "columns_test.csv"      # 输入：列清单
COLUMN_FILE_OUT = "columns_result.csv"   # 输出：列清单结果

# 批处理大小 (根据服务器内存调整，CPU 建议 256-512)
BATCH_SIZE = 512

# ================= 2. 行业通用缩写/同义词典 =================
# 策略：如果英文命中 Key，且中文命中 Value 中的任意一个，则给予额外加分。
# 这解决了 "amt" (缩写) 和 "金额" (全称) 在纯语义模型中分值不够高的问题。
COMMON_SYNONYMS = {
    # --- 核心标识 ---
    'id':    ['编号', '代码', '标识', '序号', 'id'],
    'no':    ['编号', '号码', '序号'],
    'code':  ['编码', '代码', '码'],
    'num':   ['数量', '次数', '号'],
    'nm':    ['名称', '姓名'],
    'name':  ['名称', '姓名'],
    
    # --- 金额与交易 (金融核心) ---
    'amt':   ['金额', '费用', '钱'],
    'amount':['金额', '数量'],
    'bal':   ['余额', '差额'],
    'price': ['价格', '单价'],
    'cost':  ['成本', '费用'],
    'rate':  ['利率', '汇率', '比例'],
    'txn':   ['交易', '流水'],
    'trans': ['交易', '传输'],
    'pay':   ['支付', '付款'],
    
    # --- 常用实体 ---
    'org':   ['机构', '组织', '部门'],
    'dept':  ['部门', '科室'],
    'cust':  ['客户'],
    'user':  ['用户'],
    'emp':   ['员工', '人员'],
    'mgr':   ['经理', '管理'],
    'acct':  ['账户', '账号'],
    
    # --- 时间与状态 ---
    'dt':    ['日期'],
    'date':  ['日期'],
    'tm':    ['时间'],
    'time':  ['时间', '时分'],
    'ts':    ['时间戳'],
    'stat':  ['状态', '情况'],
    'status':['状态', '情况'],
    'flg':   ['标志', '标识', '是否'],
    'flag':  ['标志', '标识', '是否'],
    'is':    ['是否'],
    'curr':  ['币种', '当前'],
    
    # --- 通用术语 ---
    'desc':  ['描述', '说明', '备注'],
    'rem':   ['备注', '摘要'],
    'remark':['备注', '摘要', '说明'],
    'addr':  ['地址'],
    'tel':   ['电话'],
    'mobile':['手机', '移动电话'],
    'msg':   ['消息', '信息'],
    'err':   ['错误', '异常'],
    'seq':   ['序号', '序列']
}

# ================= 3. 核心工具函数 =================

def download_model_if_needed():
    """
    智能下载：只下载 PyTorch 权重文件，过滤掉 TF/ONNX 等无用大文件。
    将下载量从 3GB+ 降低到 ~470MB。
    """
    if not os.path.exists(LOCAL_MODEL_DIR):
        print(f"[{MODEL_REPO}] 本地模型不存在，开始下载...")
        print("提示：正在过滤非必要文件，仅下载核心权重 (约 470MB)...")
        try:
            snapshot_download(
                repo_id=MODEL_REPO, 
                local_dir=LOCAL_MODEL_DIR,
                # 严格过滤，只下这些后缀
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.md"]
            )
            print("✅ 下载完成。")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            print("请检查网络，或尝试配置 HF_ENDPOINT 镜像源。")
            sys.exit(1)
    else:
        print(f"✅ 检测到本地模型目录 {LOCAL_MODEL_DIR}，跳过下载。")

def preprocess_text(text):
    """
    标准化清洗：将代码命名转换为自然语言。
    例如: 'isDeleted' -> 'is deleted', 'user_id' -> 'user id'
    """
    if pd.isna(text): return ""
    text = str(text)
    # 替换常见分隔符
    text = text.replace('_', ' ').replace('-', ' ')
    # 拆分驼峰命名 (在大写字母前加空格)
    text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    # 转小写并去首尾空格
    return text.lower().strip()

def check_synonym_bonus(en_text, cn_text):
    """
    规则引擎：检查是否命中行业缩写字典。
    如果命中，返回奖励分 (25分)。
    """
    # 将英文拆分为单词列表 (例如 'cust_id' -> ['cust', 'id'])
    en_words = preprocess_text(en_text).split()
    cn_text = str(cn_text)
    
    for word in en_words:
        # 如果这个单词在字典里
        if word in COMMON_SYNONYMS:
            # 检查对应的中文关键词是否出现在中文名里
            for cn_keyword in COMMON_SYNONYMS[word]:
                if cn_keyword in cn_text:
                    # 命中！比如 found 'id' and '编号'
                    return 25 
    return 0

def calculate_and_fill(df, type_name, model):
    """
    主计算逻辑：去重 -> 向量化 -> 规则修正 -> 还原
    """
    print(f"\n正在处理 {type_name} (原始行数: {len(df)})...")
    
    if type_name == 'table':
        col_en, col_cn = 'Table Name', '表中文名'
    else:
        col_en, col_cn = 'Column Name', '字段中文名'

    # 1. 健壮性检查
    if col_en not in df.columns or col_cn not in df.columns:
        print(f"⚠️ 跳过：输入文件中找不到列 {col_en} 或 {col_cn}")
        return df

    # 2. 高效去重 (Deduplication)
    # 400万行数据中，其实只有几万个唯一的单词组合，去重后计算极快
    df[col_en] = df[col_en].fillna("")
    df[col_cn] = df[col_cn].fillna("")
    
    unique_pairs = df[[col_en, col_cn]].drop_duplicates().reset_index(drop=True)
    print(f"  - 数据压缩: {len(df)} 行 -> {len(unique_pairs)} 个唯一组合")
    
    # 3. 预处理
    processed_en_list = [preprocess_text(x) for x in unique_pairs[col_en]]
    raw_cn_list = unique_pairs[col_cn].tolist()
    
    # 4. 向量化计算 (Vectorization)
    print("  - 正在进行语义向量计算 (这可能需要几分钟)...")
    # normalize_embeddings=True 之后，点积就是余弦相似度
    embeddings_en = model.encode(processed_en_list, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)
    embeddings_cn = model.encode(raw_cn_list, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)
    
    # 5. 计算基础分 (Cosine Similarity)
    print("  - 计算基础语义得分...")
    tensor_en = torch.tensor(embeddings_en)
    tensor_cn = torch.tensor(embeddings_cn)
    cosine_scores = torch.sum(tensor_en * tensor_cn, dim=1)
    base_scores = (torch.clamp(cosine_scores, 0, 1) * 100).int().tolist()
    
    # 6. 应用规则修正 (Rule-based Bonus)
    print("  - 应用行业缩写规则修正...")
    final_scores = []
    for i, score in enumerate(base_scores):
        en_raw = unique_pairs.iloc[i][col_en]
        cn_raw = unique_pairs.iloc[i][col_cn]
        
        # 计算奖励
        bonus = check_synonym_bonus(en_raw, cn_raw)
        
        # 最终得分 = 基础语义分 + 规则分，封顶 100
        # 逻辑：如果语义不通(拼音)，base分很低(30)，加了bonus也没用(30+0=30)
        # 如果语义通(缩写)，base分及格(60)，加bonus变成优秀(85)
        final_score = min(score + bonus, 100)
        final_scores.append(final_score)
        
    unique_pairs['calc_score'] = final_scores
    
    # 7. 结果映射还原 (Mapping)
    print("  - 正在将结果映射回原始数据...")
    if '关联度' in df.columns:
        df = df.drop(columns=['关联度'])
        
    result_df = pd.merge(df, unique_pairs, on=[col_en, col_cn], how='left')
    result_df = result_df.rename(columns={'calc_score': '关联度'})
    
    return result_df

# ================= 4. 主程序入口 =================

def main():
    # 1. 准备模型
    download_model_if_needed()
    
    print(f"\n正在加载模型: {LOCAL_MODEL_DIR} (CPU模式)...")
    # 加载本地模型
    model = SentenceTransformer(LOCAL_MODEL_DIR, device='cpu')
    
    # 2. 处理表清单
    if os.path.exists(TABLE_FILE_IN):
        # 使用 read_csv 读取大数据量
        df_table = pd.read_csv(TABLE_FILE_IN, dtype=str)
        df_table_result = calculate_and_fill(df_table, 'table', model)
        df_table_result.to_csv(TABLE_FILE_OUT, index=False, encoding='utf-8-sig')
        print(f"✅ 表清单处理完成: {TABLE_FILE_OUT}")
    else:
        print(f"⚠️ 未找到文件 {TABLE_FILE_IN}，跳过。")

    # 3. 处理列清单
    if os.path.exists(COLUMN_FILE_IN):
        df_col = pd.read_csv(COLUMN_FILE_IN, dtype=str)
        df_col_result = calculate_and_fill(df_col, 'column', model)
        df_col_result.to_csv(COLUMN_FILE_OUT, index=False, encoding='utf-8-sig')
        print(f"✅ 列清单处理完成: {COLUMN_FILE_OUT}")
    else:
        print(f"⚠️ 未找到文件 {COLUMN_FILE_IN}，跳过。")
        
    print("\n全部任务完成！")

if __name__ == "__main__":
    main()
