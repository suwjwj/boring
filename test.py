from transformers import BertTokenizer
from FinegrainedBERTClassifier import FinegrainedBERTClassifier
from predict import predict_sentiment
import torch

def test_model():
    # 加载模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = FinegrainedBERTClassifier()
    
    # 如果有保存的模型权重，加载它
    try:
        model.load_state_dict(torch.load('final_model.pth'))
        print("成功加载已训练的模型")
    except:
        print("未找到已训练的模型，使用未训练模型")
    
    # 测试文本
    test_texts = [
        "这个产品真的很好用，我很喜欢！",
        "服务态度太差了，非常不满意。",
        "价格还可以，质量一般。"
    ]
    
    # 进行预测
    for text in test_texts:
        result = predict_sentiment(text, model, tokenizer)
        print(f"\n文本: {result['text']}")
        print(f"情感: {result['sentiment']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("各类别概率:")
        for sentiment, prob in result['probabilities'].items():
            print(f"- {sentiment}: {prob:.4f}")

if __name__ == "__main__":
    test_model() 