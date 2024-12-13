from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re

class AdvancedSentimentAnalyzer:
    def __init__(self):
        # 使用更适合的预训练模型
        print("初始化模型中...")
        self.model_name = "nghuyong/ernie-3.0-base-zh"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        
        # 情感强度词典
        self.intensity_words = {
            'high_pos': ['非常', '很', '特别', '超级', '极其'],
            'high_neg': ['太', '极其', '非常', '特别'],
            'low_pos': ['有点', '稍微', '略微'],
            'low_neg': ['有点', '稍微', '略微']
        }
        
        # 情感转折词
        self.transition_words = ['但是', '但', '然而', '不过', '只是']
        
    def analyze_sentiment(self, text):
        """分析文本情感"""
        # 文本预处理
        text = text.strip()
        
        # 检查情感强度
        intensity = self._check_intensity(text)
        
        # 检查情感转折
        has_transition = any(word in text for word in self.transition_words)
        
        # 获取基础情感预测
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        negative_prob = probs[0][0].item()
        positive_prob = probs[0][1].item()
        
        # 根据情感强度调整概率
        if intensity == 'high':
            positive_prob = min(1.0, positive_prob * 1.2)
            negative_prob = min(1.0, negative_prob * 1.2)
        elif intensity == 'low':
            positive_prob *= 0.8
            negative_prob *= 0.8
        
        # 确定情感
        if positive_prob > 0.6:
            base_sentiment = "正面"
        elif negative_prob > 0.6:
            base_sentiment = "负面"
        else:
            base_sentiment = "中性"
            
        # 生成详细分析结果
        result = {
            "sentiment": base_sentiment,
            "positive_prob": positive_prob,
            "negative_prob": negative_prob,
            "intensity": intensity,
            "has_transition": has_transition,
            "details": []
        }
        
        # 添加分析细节
        if intensity != 'normal':
            result["details"].append(f"检测到{intensity}强度的情感表达")
        if has_transition:
            result["details"].append("存在情感转折")
            
        return result
    
    def _check_intensity(self, text):
        """检查情感强度"""
        for word in self.intensity_words['high_pos'] + self.intensity_words['high_neg']:
            if word in text:
                return 'high'
        for word in self.intensity_words['low_pos'] + self.intensity_words['low_neg']:
            if word in text:
                return 'low'
        return 'normal'

def main():
    # 初始化分析器
    analyzer = AdvancedSentimentAnalyzer()
    
    # 示例测试
    test_texts = [
        "这个产品真的很好用，我很喜欢！",
        "服务态度太差了，非常不满意。",
        "价格还可以，质量一般。",
        "虽然有点贵，但是质量确实不错。",
        "快递很快，商品完好无损。"
    ]
    
    print("\n=== 示例测试 ===")
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\n文本: {text}")
        print(f"情感: {result['sentiment']}")
        print(f"正面概率: {result['positive_prob']:.4f}")
        print(f"负面概率: {result['negative_prob']:.4f}")
        print(f"情感强度: {result['intensity']}")
        if result['has_transition']:
            print("存在情感转折")
        if result['details']:
            print("分析细节:")
            for detail in result['details']:
                print(f"- {detail}")
    
    # 交互式测试
    print("\n=== 交互式测试 ===")
    print("请输入要分析的文本（输入 'q' 退出）：")
    
    while True:
        text = input("\n>>> ")
        if text.lower() == 'q':
            break
            
        if not text.strip():
            print("请输入有效的文本！")
            continue
            
        result = analyzer.analyze_sentiment(text)
        print(f"\n情感分析结果：")
        print(f"情感: {result['sentiment']}")
        print(f"正面概率: {result['positive_prob']:.4f}")
        print(f"负面概率: {result['negative_prob']:.4f}")
        print(f"情感强度: {result['intensity']}")
        if result['has_transition']:
            print("存在情感转折")
        if result['details']:
            print("分析细节:")
            for detail in result['details']:
                print(f"- {detail}")
    
    print("\n感谢使用！再见！")

if __name__ == "__main__":
    main() 