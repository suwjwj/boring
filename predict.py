import torch
from torch.nn.functional import softmax

def predict_sentiment(text, model, tokenizer, device=None):
    """对单个文本进行情感预测"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)
    
    # 文本编码
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = softmax(outputs, dim=1)
    
    # 获取预测结果
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()
    
    # 映射情感标签
    sentiment_map = {0: '负面', 1: '正面', 2: '中性'}
    
    return {
        'text': text,
        'sentiment': sentiment_map[predicted_class],
        'confidence': confidence,
        'probabilities': {
            '负面': probs[0][0].item(),
            '正面': probs[0][1].item(),
            '中性': probs[0][2].item()
        }
    }

def batch_predict(texts, model, tokenizer, device=None):
    """批量预测多个文本"""
    return [predict_sentiment(text, model, tokenizer, device) for text in texts] 