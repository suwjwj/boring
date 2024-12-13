import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, AdamW
from sklearn.metrics import classification_report

def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def train_model(model, train_dataloader, val_dataloader, epochs=3):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        val_accuracy = evaluate_model(model, val_dataloader, device)
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Average Loss: {total_loss/len(train_dataloader):.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print("保存新的最佳模型")
        
        print('-' * 50)
    
    return model

def main():
    """主函数"""
    # 1. 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 2. 创建数据集
    df = create_sample_dataset(size=1000)  # 使用1000条数据进行训练
    
    # 3. 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 4. 创建数据集实例
    train_dataset = FinegrainedSentimentDataset(
        train_df['text'].values,
        train_df['main_emotion'].values,
        train_df['sub_emotion'].values,
        train_df['intensity'].values,
        tokenizer
    )
    
    val_dataset = FinegrainedSentimentDataset(
        val_df['text'].values,
        val_df['main_emotion'].values,
        val_df['sub_emotion'].values,
        val_df['intensity'].values,
        tokenizer
    )
    
    # 5. 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # 6. 初始化模型
    model = FinegrainedBERTClassifier()
    
    # 7. 训练模型
    trained_model = train_model(
        model, 
        train_dataloader, 
        val_dataloader,
        epochs=3
    )
    
    # 8. 保存最终模型
    torch.save(trained_model.state_dict(), 'final_model.pth')
    
    # 9. 测试模型
    test_texts = [
        "这个产品太棒了，用起来特别顺手，真是超出预期！",
        "说实话产品还行吧，但也没特别惊艳。",
        "质量很差，完全是在浪费钱，气死我了！",
        "虽然不算便宜，但是物有所值，很满意。"
    ]
    
    print("\n=== 测试结果 ===")
    for text in test_texts:
        print(f"\n原文: {text}")
        result, confidence = analyze_text(trained_model, tokenizer, text, return_confidence=True)
        print(result)
        print("预测置信度：")
        for aspect, score in confidence.items():
            print(f"- {aspect}: {score:.4f}")

if __name__ == "__main__":
    main() 