#pragma once
#include <vector>
#include <utility>

namespace nn {
// ============================================================
//  自动生成训练数据的接口（对标 sklearn.datasets）
//  返回 {data[N][D], labels[N][C]}，labels 为 one-hot 编码
// ============================================================

// 1. 高斯聚类 — 类似 sklearn.datasets.make_blobs
//    cluster_std : 类内标准差（越小越紧凑，越容易分类）
//    center_box  : 类别中心点的分布范围
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
make_blobs(int n_samples, int n_features, int n_classes,
           float cluster_std = 1.0f, float center_box = 10.0f);

// 2. 多特征分类 — 类似 sklearn.datasets.make_classification
//    包含信息特征、冗余特征、噪声特征
//    n_informative : 真正携带类别信息的特征数
//    n_redundant   : 冗余特征数（信息特征的线性组合）
//    n_clusters_per_class : 每个类的子簇数
//    class_sep     : 类别分离度（越大越容易，推荐 0.5~3.0）
//    flip_y        : 标签噪声比例 [0, 1)
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
make_classification(int n_samples, int n_features, int n_classes,
                    int n_informative, int n_redundant = 0,
                    int n_clusters_per_class = 2,
                    float class_sep = 1.0f, float flip_y = 0.0f);

// 3. 双月牙 — 经典非线性二分类（必须超过 2 维输入用不了，仅 2D）
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
make_moons(int n_samples = 100, float noise = 0.1f);

// 4. 同心圆 — 非线性二分类（仅 2D）
//    factor : 内圆半径比例
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
make_circles(int n_samples = 100, float factor = 0.5f, float noise = 0.05f);

// 5. XOR 棋盘格 — 2^dim 类，排列在超立方体顶点
//    适合测试深层网络对非线性边界的拟合能力
//    噪声越大越难分
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
make_xor_grid(int n_samples = 100, int dim = 2, float noise = 0.15f);

} // namespace nn