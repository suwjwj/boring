import torch
from transformers import BertTokenizer
import numpy as np

def get_word_importance(text, model, tokenizer):
    """计算每个词对预测结果的重要性"""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取原始预测
    encoded = tokenizer(text, return_tensors='pt')
    original_output = model(
        encoded['input_ids'].to(device),
        encoded['attention_mask'].to(device)
    )
    original_prob = torch.softmax(original_output, dim=1)
    
    # 计算每个词的重要性
    tokens = tokenizer.tokenize(text)
    importance_scores = []
    
    for i in range(len(tokens)):
        # 创建掩码版本的文本
        masked_text = tokens.copy()
        masked_text[i] = '[MASK]'
        masked_text = tokenizer.convert_tokens_to_string(masked_text)
        
        # 获取掩码后的预测
        masked_encoded = tokenizer(masked_text, return_tensors='pt')
        masked_output = model(
            masked_encoded['input_ids'].to(device),
            masked_encoded['attention_mask'].to(device)
        )
        masked_prob = torch.softmax(masked_output, dim=1)
        
        # 计算重要性分数
        importance = torch.norm(original_prob - masked_prob).item()
        importance_scores.append(importance)
    
    return list(zip(tokens, importance_scores)) 