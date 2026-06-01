#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cassert>
#include "tools.h"
namespace nn {

// ==================== 全局随机引擎 ====================
static std::mt19937& rng() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

// ---- 均匀分布 ----
static float randf(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(rng());
}

// ---- 标准正态 ----
static float randn(float mean, float stddev) {
    return std::normal_distribution<float>(mean, stddev)(rng());
}

// ==================== 辅助：生成正态随机方向向量 ====================
static std::vector<float> random_direction(int dim) {
    std::vector<float> v(dim);
    float norm = 0.0f;
    for (int i = 0; i < dim; ++i) {
        v[i] = randn(0.0f, 1.0f);
        norm += v[i] * v[i];
    }
    norm = std::sqrt(norm);
    if (norm < 1e-8f) norm = 1.0f;
    for (int i = 0; i < dim; ++i) v[i] /= norm;
    return v;
}

// ==================== 1. make_blobs — 高斯聚类 ====================
// 类似 sklearn.datasets.make_blobs
// 每个类有各自的高斯中心，样本围绕中心以 cluster_std 散布
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
make_blobs(int n_samples, int n_features, int n_classes,
           float cluster_std, float center_box) {

    std::vector<std::vector<float>> data(n_samples, std::vector<float>(n_features));
    std::vector<std::vector<float>> labels(n_samples, std::vector<float>(n_classes, 0.0f));

    // 为每个类别生成中心点（在 [-center_box, center_box] 范围内）
    std::vector<std::vector<float>> centers(n_classes);
    for (int c = 0; c < n_classes; ++c) {
        centers[c].resize(n_features);
        for (int j = 0; j < n_features; ++j)
            centers[c][j] = randf(-center_box, center_box);
    }

    for (int i = 0; i < n_samples; ++i) {
        int cls = i % n_classes;
        for (int j = 0; j < n_features; ++j)
            data[i][j] = randn(centers[cls][j], cluster_std);
        labels[i][cls] = 1.0f;
    }

    // shuffle
    std::vector<int> idx(n_samples);
    for (int i = 0; i < n_samples; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng());

    decltype(data)   sd(n_samples);
    decltype(labels) sl(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        sd[i] = data[idx[i]];
        sl[i] = labels[idx[i]];
    }
    return {sd, sl};
}

// ==================== 2. make_classification — 类似 sklearn ====================
// 核心思路：
//   - 为每个样本先生成一个低维的"信息特征"向量
//   - 每个类定义 k 个超立方体顶点作为子簇中心
//   - 样本围绕子簇中心做高斯扰动
//   - 再通过随机旋转矩阵扩展到 n_features 维
//   - 加入冗余特征（信息特征的线性组合）
//   - 加入纯噪声特征
//
// 参数说明：
//   n_samples       : 样本数
//   n_features      : 总特征维度
//   n_classes       : 类别数
//   n_informative   : 真正携带分类信息的特征数
//   n_redundant     : 冗余特征数（由信息特征线性组合生成）
//   n_clusters_per_class : 每个类的子簇数（≥1）
//   class_sep       : 类别分离度，越大越容易分（推荐 0.5~3.0）
//   flip_y          : 标签噪声比例（随机翻转标签）
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
make_classification(int n_samples, int n_features, int n_classes,
                    int n_informative, int n_redundant,
                    int n_clusters_per_class,
                    float class_sep, float flip_y) {

    assert(n_informative >= 1);
    assert(n_informative + n_redundant <= n_features);
    assert(n_clusters_per_class >= 1);
    std::cout << "Generating classification data with " << n_samples << " samples, "
              << n_features << " features (" << n_informative << " informative, "
              << n_redundant << " redundant), " << n_classes << " classes, "
              << n_clusters_per_class << " clusters/class, class_sep=" << class_sep
              << ", flip_y=" << flip_y << std::endl;
    int n_noise = n_features - n_informative - n_redundant; // 纯噪声特征数

    // ---- Step 1: 生成 n_informative 维信息特征 ----
    // 超立方体边长 ≈ class_sep
    float box = class_sep * 2.0f;

    std::vector<std::vector<float>> X_info(n_samples, std::vector<float>(n_informative));
    std::vector<int> y(n_samples);

    int total_clusters = n_classes * n_clusters_per_class;

    // 为每个子簇生成超立方体顶点作为中心
    std::vector<std::vector<float>> cluster_centers(total_clusters,
                                                     std::vector<float>(n_informative));
    for (int c = 0; c < total_clusters; ++c) {
        for (int j = 0; j < n_informative; ++j)
            cluster_centers[c][j] = randf(-box, box);
    }

    for (int i = 0; i < n_samples; ++i) {
        int cls   = i % n_classes;
        int sub   = (i / n_classes) % n_clusters_per_class;
        int c_idx = cls * n_clusters_per_class + sub;

        for (int j = 0; j < n_informative; ++j)
            X_info[i][j] = randn(cluster_centers[c_idx][j], 1.0f);

        y[i] = cls;
    }

    // ---- Step 2: 随机旋转 + 缩放，映射到 n_informative 维空间 ----
    // （生成随机旋转矩阵 Q：用随机高斯矩阵的 QR 分解的 Q）
    {
        // 生成 n_informative × n_informative 的随机高斯矩阵
        std::vector<std::vector<float>> M(n_informative, std::vector<float>(n_informative));
        for (int i = 0; i < n_informative; ++i)
            for (int j = 0; j < n_informative; ++j)
                M[i][j] = randn(0.0f, 1.0f);

        // 用 Gram-Schmidt 做正交化得到旋转矩阵 Q
        std::vector<std::vector<float>> Q(n_informative, std::vector<float>(n_informative));
        for (int col = 0; col < n_informative; ++col) {
            // 复制第 col 列
            for (int row = 0; row < n_informative; ++row)
                Q[row][col] = M[row][col];
            // 减去在前 col 列上的投影
            for (int prev = 0; prev < col; ++prev) {
                float dot = 0.0f;
                for (int row = 0; row < n_informative; ++row)
                    dot += Q[row][col] * Q[row][prev];
                for (int row = 0; row < n_informative; ++row)
                    Q[row][col] -= dot * Q[row][prev];
            }
            // 归一化
            float norm = 0.0f;
            for (int row = 0; row < n_informative; ++row)
                norm += Q[row][col] * Q[row][col];
            norm = std::sqrt(norm);
            if (norm > 1e-8f) {
                for (int row = 0; row < n_informative; ++row)
                    Q[row][col] /= norm;
            }
        }

        // 对每个样本 X_info[i] 应用旋转: X_info_rot[i] = Q * X_info[i]
        for (int i = 0; i < n_samples; ++i) {
            std::vector<float> rotated(n_informative, 0.0f);
            for (int row = 0; row < n_informative; ++row)
                for (int col = 0; col < n_informative; ++col)
                    rotated[row] += Q[row][col] * X_info[i][col];
            X_info[i] = std::move(rotated);
        }
    }

    // ---- Step 3: 拼装完整特征矩阵 ----
    std::vector<std::vector<float>> X(n_samples, std::vector<float>(n_features));
    std::vector<std::vector<float>> labels(n_samples, std::vector<float>(n_classes, 0.0f));

    for (int i = 0; i < n_samples; ++i) {
        // a) 信息特征
        for (int j = 0; j < n_informative; ++j)
            X[i][j] = X_info[i][j];

        // b) 冗余特征 — 信息特征的随机线性组合 + 噪声
        for (int j = 0; j < n_redundant; ++j) {
            float val = 0.0f;
            for (int k = 0; k < n_informative; ++k)
                val += randn(0.0f, 1.0f) * X_info[i][k];
            val += randn(0.0f, 0.1f); // 小噪声
            X[i][n_informative + j] = val;
        }

        // c) 纯噪声特征
        for (int j = 0; j < n_noise; ++j)
            X[i][n_informative + n_redundant + j] = randn(0.0f, 1.0f);

        // 标签
        int cls = y[i];
        // 按 flip_y 概率随机翻转
        if (randf(0.0f, 1.0f) < flip_y) {
            int new_cls;
            do { new_cls = (int)randf(0, (float)n_classes); } while (new_cls == cls);
            cls = new_cls;
        }
        labels[i][cls] = 1.0f;
    }

    // ---- Step 4: shuffle ----
    std::vector<int> idx(n_samples);
    for (int i = 0; i < n_samples; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng());

    decltype(X)      sX(n_samples);
    decltype(labels) sL(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        sX[i] = X[idx[i]];
        sL[i] = labels[idx[i]];
    }
    return {sX, sL};
}

// ==================== 3. make_moons — 双月牙（非线性二分类） ====================
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
make_moons(int n_samples, float noise) {
    std::vector<std::vector<float>> data(n_samples, std::vector<float>(2));
    std::vector<std::vector<float>> labels(n_samples, std::vector<float>(2, 0.0f));

    int n0 = n_samples / 2;
    int n1 = n_samples - n0;

    // 上弦月（class 0）
    for (int i = 0; i < n0; ++i) {
        float t = randf(0.0f, 3.141592653589793f);
        data[i][0] = std::cos(t) + randn(0.0f, noise);
        data[i][1] = std::sin(t) + randn(0.0f, noise);
        labels[i][0] = 1.0f;
    }

    // 下弦月（class 1），向下平移
    for (int i = 0; i < n1; ++i) {
        float t = randf(0.0f, 3.141592653589793f);
        data[n0 + i][0] = 1.0f - std::cos(t) + randn(0.0f, noise);
        data[n0 + i][1] = 1.0f - std::sin(t) - 0.5f + randn(0.0f, noise);
        labels[n0 + i][1] = 1.0f;
    }

    // shuffle
    std::vector<int> idx(n_samples);
    for (int i = 0; i < n_samples; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng());

    decltype(data)   sd(n_samples);
    decltype(labels) sl(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        sd[i] = data[idx[i]];
        sl[i] = labels[idx[i]];
    }
    return {sd, sl};
}

// ==================== 4. make_circles — 同心圆（非线性二分类） ====================
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
make_circles(int n_samples, float factor, float noise) {
    std::vector<std::vector<float>> data(n_samples, std::vector<float>(2));
    std::vector<std::vector<float>> labels(n_samples, std::vector<float>(2, 0.0f));

    int n0 = n_samples / 2;
    int n1 = n_samples - n0;

    // 外圆（class 0），半径 1.0
    for (int i = 0; i < n0; ++i) {
        float angle = randf(0.0f, 2.0f * 3.141592653589793f);
        float r = 1.0f + randn(0.0f, noise);
        data[i][0] = r * std::cos(angle);
        data[i][1] = r * std::sin(angle);
        labels[i][0] = 1.0f;
    }

    // 内圆（class 1），半径 factor
    for (int i = 0; i < n1; ++i) {
        float angle = randf(0.0f, 2.0f * 3.141592653589793f);
        float r = factor + randn(0.0f, noise);
        data[n0 + i][0] = r * std::cos(angle);
        data[n0 + i][1] = r * std::sin(angle);
        labels[n0 + i][1] = 1.0f;
    }

    // shuffle
    std::vector<int> idx(n_samples);
    for (int i = 0; i < n_samples; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng());

    decltype(data)   sd(n_samples);
    decltype(labels) sl(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        sd[i] = data[idx[i]];
        sl[i] = labels[idx[i]];
    }
    return {sd, sl};
}

// ==================== 5. make_xor_grid — XOR 棋盘格（多分类非线性） ====================
// 生成 2^dim 个类，排列在超立方体顶点上；每类再加高斯噪声
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
make_xor_grid(int n_samples, int dim, float noise) {
    int n_classes = 1 << dim;  // 2^dim

    std::vector<std::vector<float>> data(n_samples, std::vector<float>(dim));
    std::vector<std::vector<float>> labels(n_samples, std::vector<float>(n_classes, 0.0f));

    for (int i = 0; i < n_samples; ++i) {
        int cls = i % n_classes;
        // 把 cls 的二进制位映射到每个维度的 ±1 坐标
        for (int j = 0; j < dim; ++j) {
            float sign = ((cls >> j) & 1) ? 1.0f : -1.0f;
            data[i][j] = sign + randn(0.0f, noise);
        }
        labels[i][cls] = 1.0f;
    }

    std::vector<int> idx(n_samples);
    for (int i = 0; i < n_samples; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng());

    decltype(data)   sd(n_samples);
    decltype(labels) sl(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        sd[i] = data[idx[i]];
        sl[i] = labels[idx[i]];
    }
    return {sd, sl};
}

} // namespace nn