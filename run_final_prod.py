import os
import shutil
import sys
import re
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

# ================= 1. 全局参数配置 =================
# 模型仓库名称 (多语言版 MiniLM)
MODEL_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 本地保存路径
LOCAL_MODEL_DIR = "./multilingual-minilm-local"

# 输入输出文件配置
TABLE_FILE_IN = "tables_test.csv"
TABLE_FILE_OUT = "tables_result.csv"
COLUMN_FILE_IN = "columns_test.csv"
COLUMN_FILE_OUT = "columns_result.csv"

# 批处理大小 (根据内存调整)
BATCH_SIZE = 512

# ================= 2. 行业专家词典 (规则引擎) =================
# 作用：英文缩写命中 Key 且 中文包含 Value -> 强制加 25 分
COMMON_SYNONYMS = {
    # --- 核心标识 ---
    'id':    ['编号', '代码', '标识', '序号', 'id'],
    'no':    ['编号', '号码', '序号'],
    'code':  ['编码', '代码', '码'],
    'num':   ['数量', '次数', '号'],
    'nm':    ['名称', '姓名'],
    'name':  ['名称', '姓名'],
    
    # --- 金额与交易 ---
    'amt':   ['金额', '费用', '钱'],
    'amount':['金额', '数量'],
    'bal':   ['余额', '差额'],
    'price': ['价格', '单价'],
    'cost':  ['成本', '费用'],
    'rate':  ['利率', '汇率', '比例'],
    'txn':   ['交易', '流水'],
    'trans': ['交易', '传输'],
    'pay':   ['支付', '付款'],
    
    # --- 组织与人员 ---
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
    
    # --- 通用 ---
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

# ================= 3. 智能下载与校验模块 =================

def check_model_integrity():
    """
    检查模型核心文件是否存在。
    适配 sentence-transformers 2.x 版本，必须检查 pytorch_model.bin
    """
    # 必须包含: 权重文件, 主配置, 分词配置
    required_files = ["pytorch_model.bin", "config.json", "tokenizer.json", "sentencepiece.bpe.model"]
    
    if not os.path.exists(LOCAL_MODEL_DIR):
        return False
    
    # 简单的存在性检查
    for f in required_files:
        if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, f)):
            # 兼容性检查：有些文件可能在子目录，这里主要检查根目录关键文件
            if f == "sentencepiece.bpe.model": continue 
            return False
    return True

def download_model_smartly():
    """
    智能下载逻辑 (官方源 + 兼容旧版库)：
    1. 连接 Hugging Face 官方服务器。
    2. 下载 *.json (解决 1_Pooling/config.json 缺失报错)。
    3. 下载 *.model (解决分词器报错)。
    4. 下载 pytorch_model.bin (适配 sentence-transformers 2.2.2)。
    """
    if check_model_integrity():
        print(f"✅ 检测到完整模型: {LOCAL_MODEL_DIR}，跳过下载。")
        return

    # 清理残损目录
    if os.path.exists(LOCAL_MODEL_DIR):
        print("⚠️ 检测到目录不完整，正在清理并重新下载...")
        try:
            shutil.rmtree(LOCAL_MODEL_DIR)
        except Exception as e:
            print(f"❌ 清理失败: {e} (请手动删除文件夹)")

    print(f"⬇️ 正在从官方源下载模型: {MODEL_REPO} ...")
    print("   (模式：仅下载 PyTorch 权重和必要配置，约 470MB)")
    
    try:
        snapshot_download(
            repo_id=MODEL_REPO, 
            local_dir=LOCAL_MODEL_DIR,
            # 【关键配置】
            # 必须包含 *.json (为了下载子文件夹里的配置)
            # 必须包含 pytorch_model.bin (兼容性最佳)
            allow_patterns=[
                "*.json", 
                "*.txt", 
                "*.model", 
                "pytorch_model.bin", 
                "README.md"
            ],
            # 坚决不下载这些大文件
            ignore_patterns=["*.safetensors", "*.onnx", "*.h5", "openvino*", "*.msgpack"],
            resume_download=True
        )
        print("✅ 下载成功！")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("提示：官方源连接超时，请检查网络或代理设置。")
        sys.exit(1)

# ================= 4. 数据处理核心逻辑 =================

def preprocess_text(text):
    """文本清洗：驼峰拆分、去符、转小写"""
    if pd.isna(text): return ""
    text = str(text)
    text = text.replace('_', ' ').replace('-', ' ')
    # 拆分驼峰 (e.g., 'isDeleted' -> 'is Deleted')
    text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
    return text.lower().strip()

def get_synonym_bonus(en_text, cn_text):
    """计算规则奖励分"""
    en_words = preprocess_text(en_text).split()
    cn_text = str(cn_text)
    
    for word in en_words:
        if word in COMMON_SYNONYMS:
            for cn_keyword in COMMON_SYNONYMS[word]:
                if cn_keyword in cn_text:
                    return 25 # 命中规则，奖励 25 分
    return 0

def process_file(file_in, file_out, type_name, model):
    """通用文件处理流程：读取 -> 计算 -> 保存"""
    if not os.path.exists(file_in):
        print(f"⚠️ 跳过：找不到输入文件 {file_in}")
        return

    print(f"\n🚀 正在处理 {type_name} ...")
    try:
        df = pd.read_csv(file_in, dtype=str)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    if type_name == 'table':
        col_en, col_cn = 'Table Name', '表中文名'
    else:
        col_en, col_cn = 'Column Name', '字段中文名'

    if col_en not in df.columns or col_cn not in df.columns:
        print(f"❌ 列名错误：文件中必须包含 '{col_en}' 和 '{col_cn}'")
        return

    # 1. 高效去重
    print(f"   - 原始数据: {len(df)} 行")
    df[col_en] = df[col_en].fillna("")
    df[col_cn] = df[col_cn].fillna("")
    
    unique_pairs = df[[col_en, col_cn]].drop_duplicates().reset_index(drop=True)
    print(f"   - 去重后需计算: {len(unique_pairs)} 行")

    # 2. 向量化
    processed_en = [preprocess_text(x) for x in unique_pairs[col_en]]
    raw_cn = unique_pairs[col_cn].tolist()

    print("   - 正在计算 AI 语义向量 (CPU)...")
    embeddings_en = model.encode(processed_en, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)
    embeddings_cn = model.encode(raw_cn, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)

    # 3. 评分
    print("   - 正在计算综合得分...")
    tensor_en = torch.tensor(embeddings_en)
    tensor_cn = torch.tensor(embeddings_cn)
    
    cosine_scores = torch.sum(tensor_en * tensor_cn, dim=1)
    base_scores = (torch.clamp(cosine_scores, 0, 1) * 100).int().tolist()

    final_scores = []
    for i, score in enumerate(base_scores):
        en_raw = unique_pairs.iloc[i][col_en]
        cn_raw = unique_pairs.iloc[i][col_cn]
        bonus = get_synonym_bonus(en_raw, cn_raw)
        final_scores.append(min(score + bonus, 100))

    unique_pairs['calc_score'] = final_scores

    # 4. 还原保存
    if '关联度' in df.columns:
        df = df.drop(columns=['关联度'])
        
    result_df = pd.merge(df, unique_pairs, on=[col_en, col_cn], how='left')
    result_df = result_df.rename(columns={'calc_score': '关联度'})
    
    result_df.to_csv(file_out, index=False, encoding='utf-8-sig')
    print(f"✅ 完成！已保存: {file_out}")

# ================= 5. 主程序入口 =================

def main():
    print("="*50)
    print("      数据治理 AI 映射工具 (官方源兼容版)      ")
    print("="*50)

    # 1. 下载/检查模型
    download_model_smartly()
    
    # 2. 加载模型
    print(f"\n正在加载模型: {LOCAL_MODEL_DIR} ...")
    try:
        model = SentenceTransformer(LOCAL_MODEL_DIR, device='cpu')
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("建议：删除目录 ./multilingual-minilm-local 后重试。")
        return

    # 3. 执行任务
    process_file(TABLE_FILE_IN, TABLE_FILE_OUT, 'table', model)
    process_file(COLUMN_FILE_IN, COLUMN_FILE_OUT, 'column', model)

    print("\n🎉 全部结束。")

if __name__ == "__main__":
    main()
