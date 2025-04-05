class SimilarityAttentionFusionLayer(nn.Module):
    def __init__(self):
        super(SimilarityAttentionFusionLayer, self).__init__()
        self.fusion = nn.Linear(768 * 3, 768)

    def forward(self, content_cls_token, analyze_cls_token, comments_cls_token):
        # 计算每对特征之间的余弦相似度
        content_analyze_similarity = F.cosine_similarity(content_cls_token, analyze_cls_token, dim=1).unsqueeze(1)
        content_comments_similarity = F.cosine_similarity(content_cls_token, comments_cls_token, dim=1).unsqueeze(1)
        analyze_comments_similarity = F.cosine_similarity(analyze_cls_token, comments_cls_token, dim=1).unsqueeze(1)

        # 计算相似度的权重（基于相似度的注意力机制）
        similarity_weights = torch.cat((content_analyze_similarity, content_comments_similarity, analyze_comments_similarity), dim=1)
        similarity_weights = F.softmax(similarity_weights, dim=1)  # Softmax确保权重和为1

        # 使用相似度权重对特征进行加权求和
        weighted_content = similarity_weights[:, 0].unsqueeze(1) * content_cls_token
        weighted_analyze = similarity_weights[:, 1].unsqueeze(1) * analyze_cls_token
        weighted_comments = similarity_weights[:, 2].unsqueeze(1) * comments_cls_token

        # 合并加权后的特征
        fused_features = torch.cat((weighted_content, weighted_analyze, weighted_comments), dim=1)  # [batch_size, 768 * 3]
        fused_features = self.fusion(fused_features)  # [batch_size, 768]
        return fused_features

class AttentionFusionLayer(nn.Module):
    def __init__(self):
        super(AttentionFusionLayer, self).__init__()
        self.fusion = nn.Linear(768 * 3, 768)

        # 注意力权重生成器
        self.attention_weights = nn.Linear(768*3, 3)  # 生成三个权重

    def forward(self, content_cls_token, analyze_cls_token, comments_cls_token):
        # 合并输入特征
        combined_features = torch.cat((content_cls_token, analyze_cls_token, comments_cls_token), dim=1)  # [batch_size, 768 * 3]

        # 计算注意力权重
        attention_scores = self.attention_weights(combined_features)  # [batch_size, 3]
        attention_weights = F.softmax(attention_scores, dim=1)  # Softmax归一化，确保权重和为1

        # 使用注意力权重加权特征
        weighted_content = attention_weights[:, 0].unsqueeze(1) * content_cls_token
        weighted_analyze = attention_weights[:, 1].unsqueeze(1) * analyze_cls_token
        weighted_comments = attention_weights[:, 2].unsqueeze(1) * comments_cls_token

        # 合并加权后的特征
        fused_features = self.fusion(torch.cat((weighted_content, weighted_analyze, weighted_comments), dim=1))  # [batch_size, 768]
        return fused_features

class GatedFusionLayer(nn.Module):
    def __init__(self):
        super(GatedFusionLayer, self).__init__()
        # 门控生成器
        self.gate_content = nn.Linear(768, 1)
        self.gate_analyze = nn.Linear(768, 1)
        self.gate_comments = nn.Linear(768, 1)

        # 融合层
        self.fusion = nn.Linear(768 * 3, 768)

    def forward(self, content_cls_token, analyze_cls_token, comments_cls_token):
        # 计算每个输入的门控值
        gate_content = torch.sigmoid(self.gate_content(content_cls_token))  # [batch_size, 1]
        gate_analyze = torch.sigmoid(self.gate_analyze(analyze_cls_token))  # [batch_size, 1]
        gate_comments = torch.sigmoid(self.gate_comments(comments_cls_token))  # [batch_size, 1]

        # 门控加权特征
        gated_content = gate_content * content_cls_token
        gated_analyze = gate_analyze * analyze_cls_token
        gated_comments = gate_comments * comments_cls_token

        # 合并加权后的特征
        fused_features = self.fusion(torch.cat((gated_content, gated_analyze, gated_comments), dim=1))  # [batch_size, 768]
        return fused_features


class SLMModel(nn.Module):
    def __init__(self, bert_model, hidden_size=256, num_labels=2):
        super(SLMModel, self).__init__()
        self.bert = bert_model
        # 冻结BERT参数, 微调最后一层
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        # 定义融合层
        self.similarity_fusion_layer = SimilarityAttentionFusionLayer()
        self.attention_fusion_layer = AttentionFusionLayer()
        self.gated_fusion_layer = GatedFusionLayer()
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(768*3, hidden_size),  # 768 输入维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, content_input_ids, content_attention_mask, 
                analyze_input_ids, analyze_attention_mask, 
                comments_input_ids, comments_attention_mask):
        # 提取BERT特征
        content_outputs = self.bert(input_ids=content_input_ids, attention_mask=content_attention_mask)
        analyze_outputs = self.bert(input_ids=analyze_input_ids, attention_mask=analyze_attention_mask)
        comments_outputs = self.bert(input_ids=comments_input_ids, attention_mask=comments_attention_mask)

        # 获取 last_hidden_state
        content_last_hidden_state = content_outputs.last_hidden_state
        analyze_last_hidden_state = analyze_outputs.last_hidden_state
        comments_last_hidden_state = comments_outputs.last_hidden_state
        
        # 调整维度: 只取 [CLS] 标记的输出
        content_last_hidden_state = content_last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_dim]
        analyze_last_hidden_state = analyze_last_hidden_state[:, 0, :]
        comments_last_hidden_state = comments_last_hidden_state[:, 0, :]

        # 使用融合层
        fused_features_sim = self.similarity_fusion_layer(content_last_hidden_state, analyze_last_hidden_state, comments_last_hidden_state)
        fused_features_att = self.attention_fusion_layer(content_last_hidden_state, analyze_last_hidden_state, comments_last_hidden_state)
        fused_features_gated = self.gated_fusion_layer(content_last_hidden_state, analyze_last_hidden_state, comments_last_hidden_state)
        # 合并三种融合方式的特征
        fused_features = torch.cat((fused_features_sim, fused_features_att,fused_features_gated), dim=1)
        # 分类层
        logits = self.classifier(fused_features)

        return logits