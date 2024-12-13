from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import jieba
import jieba.posseg as pseg

class SentimentAnalyzer:
    def __init__(self):
        print("初始化模型中...")
        # 使用经过情感分析微调的模型
        self.model_name = "uer/roberta-base-finetuned-jd-binary-chinese"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        
        # 定义语义结构
        self.semantic_patterns = {
            "转折关系": ["但", "但是", "却", "然而", "不过", "只是"],
            "递进关系": ["而且", "并且", "不仅", "甚至"],
            "条件关系": ["如果", "虽然", "尽管", "即使"],
            "因果关系": ["因为", "所以", "因此", "由于"]
        }
        
        # 定义情感词典
        self.sentiment_words = {
            "positive": ["好", "喜欢", "优秀", "棒", "满意", "开心", "快乐"],
            "negative": ["差", "糟", "烂", "失望", "不满", "难过", "伤心"],
            "degree": ["很", "非常", "特别", "极其", "有点", "稍微"]
        }
    
    def _analyze_structure(self, text):
        """分析句子结构"""
        structure_info = {"type": None, "parts": []}
        
        # 检查各种语义关系
        for relation_type, markers in self.semantic_patterns.items():
            for marker in markers:
                if marker in text:
                    parts = text.split(marker)
                    if len(parts) == 2:
                        return {
                            "type": relation_type,
                            "marker": marker,
                            "parts": [p.strip() for p in parts]
                        }
        
        return {"type": "单句", "parts": [text]}
    
    def _analyze_sentiment_words(self, text):
        """分析情感词"""
        words = jieba.lcut(text)
        sentiment_count = {"positive": 0, "negative": 0}
        degree_words = []
        
        for i, word in enumerate(words):
            # 检查情感词
            if word in self.sentiment_words["positive"]:
                sentiment_count["positive"] += 1
            elif word in self.sentiment_words["negative"]:
                sentiment_count["negative"] += 1
            # 检查程度词
            elif word in self.sentiment_words["degree"]:
                degree_words.append(word)
        
        return sentiment_count, degree_words
    
    def analyze(self, text):
        """分析文本情感"""
        # 1. 分析句子结构
        structure = self._analyze_structure(text)
        
        # 2. 获取模型预测的基础概率
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        base_negative_prob = probs[0][0].item()
        base_positive_prob = probs[0][1].item()
        
        # 3. 分析情感词
        sentiment_count, degree_words = self._analyze_sentiment_words(text)
        
        # 4. 根据结构和情感词调整概率
        adjusted_positive_prob = base_positive_prob
        adjusted_negative_prob = base_negative_prob
        
        # 4.1 结构调整
        if structure["type"] != "单句":
            parts = structure["parts"]
            # 分析前后句的情感词
            front_sentiment, _ = self._analyze_sentiment_words(parts[0])
            back_sentiment, _ = self._analyze_sentiment_words(parts[1])
            
            if structure["type"] == "转折关系":
                # 转折关系主要看后半句，权重0.7
                weight = 0.7
                if back_sentiment["positive"] > back_sentiment["negative"]:
                    adjusted_positive_prob = base_positive_prob * weight + (1 - weight) * base_negative_prob
                    adjusted_negative_prob = base_negative_prob * (1 - weight)
                elif back_sentiment["negative"] > back_sentiment["positive"]:
                    adjusted_negative_prob = base_negative_prob * weight + (1 - weight) * base_positive_prob
                    adjusted_positive_prob = base_positive_prob * (1 - weight)
            
            elif structure["type"] == "递进关系":
                # 递进关系加强语气
                if front_sentiment["positive"] > 0 and back_sentiment["positive"] > 0:
                    adjusted_positive_prob = min(1.0, base_positive_prob * 1.3)
                    adjusted_negative_prob = max(0.0, base_negative_prob * 0.7)
                elif front_sentiment["negative"] > 0 and back_sentiment["negative"] > 0:
                    adjusted_negative_prob = min(1.0, base_negative_prob * 1.3)
                    adjusted_positive_prob = max(0.0, base_positive_prob * 0.7)
        
        # 4.2 情感词调整
        sentiment_weight = 0.3
        if sentiment_count["positive"] > sentiment_count["negative"]:
            word_factor = sentiment_count["positive"] / (sentiment_count["positive"] + sentiment_count["negative"] + 1)
            adjusted_positive_prob = adjusted_positive_prob * (1 - sentiment_weight) + word_factor * sentiment_weight
        elif sentiment_count["negative"] > sentiment_count["positive"]:
            word_factor = sentiment_count["negative"] / (sentiment_count["positive"] + sentiment_count["negative"] + 1)
            adjusted_negative_prob = adjusted_negative_prob * (1 - sentiment_weight) + word_factor * sentiment_weight
        
        # 4.3 程度词调整
        if degree_words:
            intensity = min(len(degree_words) * 0.1, 0.3)  # 最多增强30%
            if adjusted_positive_prob > adjusted_negative_prob:
                adjusted_positive_prob = min(1.0, adjusted_positive_prob * (1 + intensity))
            else:
                adjusted_negative_prob = min(1.0, adjusted_negative_prob * (1 + intensity))
        
        # 5. 确保概率和为1
        total_prob = adjusted_positive_prob + adjusted_negative_prob
        adjusted_positive_prob /= total_prob
        adjusted_negative_prob /= total_prob
        
        # 6. 确定最终情感
        if adjusted_positive_prob > 0.6:
            sentiment = "正面"
        elif adjusted_negative_prob > 0.6:
            sentiment = "负面"
        else:
            sentiment = "中性"
        
        # 7. 生成分析结果
        result = {
            "text": text,
            "sentiment": sentiment,
            "positive_prob": adjusted_positive_prob,
            "negative_prob": adjusted_negative_prob,
            "base_positive_prob": base_positive_prob,
            "base_negative_prob": base_negative_prob,
            "structure": structure["type"],
            "sentiment_words": sentiment_count,
            "degree_words": len(degree_words)
        }
        
        return result

def main():
    # 初始化分析器
    analyzer = SentimentAnalyzer()
    
    # 示例测试
    test_texts = [
        "这个产品真的很好用，我很喜欢！",
        "服务态度太差了，非常不满意。",
        "价格还可以，质量一般。",
        "虽然有点贵，但是质量确实不错。",
        "本来很期待，结果让人失望。",
        "不仅价格合理，而且质量很好。"
    ]
    
    print("\n=== 示例测试 ===")
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\n文本: {result['text']}")
        print(f"情感: {result['sentiment']}")
        print(f"句子结构: {result['structure']}")
        print(f"正面概率: {result['positive_prob']:.4f}")
        print(f"负面概率: {result['negative_prob']:.4f}")
        print(f"情感词统计: 正面({result['sentiment_words']['positive']}), " 
              f"负面({result['sentiment_words']['negative']})")
        print(f"程度词数量: {result['degree_words']}")
    
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
            
        result = analyzer.analyze(text)
        print(f"\n情感分析结果：")
        print(f"情感: {result['sentiment']}")
        print(f"句子结构: {result['structure']}")
        print(f"正面概率: {result['positive_prob']:.4f}")
        print(f"负面概率: {result['negative_prob']:.4f}")
        print(f"情感词统计: 正面({result['sentiment_words']['positive']}), "
              f"负面({result['sentiment_words']['negative']})")
        print(f"程度词数量: {result['degree_words']}")
    
    print("\n感谢使用！再见！")

if __name__ == "__main__":
    main() 