import pandas as pd

def create_sample_data():
    """创建示例数据集"""
    data = {
        'text': [
            '这个产品真的很好用，我很喜欢！',
            '服务态度太差了，非常不满意。',
            '价格还可以，质量一般。',
            '非常满意的一次购物体验！',
            '快递很快，商品完好无损。'
        ],
        'label': [1, 0, 2, 1, 1]  # 1: 正面, 0: 负面, 2: 中性
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_dataset.csv', index=False)
    return df 