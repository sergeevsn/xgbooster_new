#include "xgbooster.hpp"
#include <QDebug>

static void safe_xgboost(int call) {
    if (call != 0) {
        throw std::runtime_error(XGBGetLastError());
    }
}

XGBModel::XGBModel(const QMap<QString, QString>& params, QObject* parent)
    : QObject(parent), params_(params) {}

XGBModel::~XGBModel() {
    if (dtrain_) XGDMatrixFree(dtrain_);
    if (booster_) XGBoosterFree(booster_);
}

void XGBModel::CreateDMatrix(const QVector<QVector<double>>& X, DMatrixHandle& dmat) {
    size_t n_rows = X.size();
    if (n_rows == 0)
        throw std::invalid_argument("Empty feature matrix");

    n_features_ = X[0].size();

    QVector<float> flat_X;
    flat_X.reserve(n_rows * n_features_);
    for (const auto& row : X) {
        if (row.size() != n_features_)
            throw std::invalid_argument("Inconsistent feature size");
        for (double val : row)
            flat_X.append(static_cast<float>(val));
    }

    safe_xgboost(XGDMatrixCreateFromMat(flat_X.data(), n_rows, n_features_, -1, &dmat));
}

void XGBModel::SetBoosterParams() {
    for (auto it = params_.begin(); it != params_.end(); ++it) {
        safe_xgboost(XGBoosterSetParam(booster_, it.key().toUtf8().constData(), it.value().toUtf8().constData()));
    }
}

void XGBModel::SaveModel(const QString& filename) {
    safe_xgboost(XGBoosterSaveModel(booster_, filename.toUtf8().constData()));
}

void XGBModel::LoadModel(const QString& filename) {
    safe_xgboost(XGBoosterCreate(nullptr, 0, &booster_));
    safe_xgboost(XGBoosterLoadModel(booster_, filename.toUtf8().constData()));
}

// ---------------------- XGBRegressor ----------------------

XGBRegressor::XGBRegressor(const QMap<QString, QString>& params, QObject* parent)
    : XGBModel(params, parent) {
    params_["objective"] = "reg:squarederror";
}

void XGBRegressor::Fit(const QVector<QVector<double>>& X,
                       const QVector<double>& y,
                       float startProgressValue,
                       float endProgressValue) {
    CreateDMatrix(X, dtrain_);

    QVector<float> y_f;
    y_f.reserve(y.size());
    for (double v : y)
        y_f.append(static_cast<float>(v));

    safe_xgboost(XGDMatrixSetFloatInfo(dtrain_, "label", y_f.data(), y.size()));
    safe_xgboost(XGBoosterCreate(&dtrain_, 1, &booster_));
    SetBoosterParams();

    int n_iter = params_.contains("num_boost_round")
        ? params_["num_boost_round"].toInt()
        : 10;

    float progressWidth = endProgressValue - startProgressValue;

    for (int i = 0; i < n_iter; ++i) {
        if (terminated_) {
            qWarning("Training was terminated by user.");
            return;
        }
        safe_xgboost(XGBoosterUpdateOneIter(booster_, i, dtrain_));
        emit progress(startProgressValue + progressWidth * float(i + 1) / n_iter);
    }
}


QVector<double> XGBRegressor::Predict(const QVector<QVector<double>>& X) {
    DMatrixHandle dtest = nullptr;
    CreateDMatrix(X, dtest);

    bst_ulong out_len = 0;
    const float* out_result = nullptr;

    safe_xgboost(XGBoosterPredict(booster_, dtest, 0, 0, 0, &out_len, &out_result));

    QVector<double> result;
    result.reserve(out_len);
    for (bst_ulong i = 0; i < out_len; ++i)
        result.append(static_cast<double>(out_result[i]));

    safe_xgboost(XGDMatrixFree(dtest));
    return result;
}


// ---------------------- XGBClassifier ----------------------

XGBClassifier::XGBClassifier(const QMap<QString, QString>& params, QObject* parent)
    : XGBModel(params, parent) {
    params_["objective"] = "multi:softmax";
}

QVector<float> XGBClassifier::EncodeLabels(const QVector<double>& y) {
    label_to_index_.clear();
    index_to_label_.clear();
    QVector<float> encoded;
    encoded.reserve(y.size());

    int idx = 0;
    for (double label : y) {
        if (!label_to_index_.contains(label)) {
            label_to_index_[label] = idx++;
            index_to_label_.append(label);
        }
        encoded.append(label_to_index_[label]);
    }
    return encoded;
}

QVector<double> XGBClassifier::DecodeLabels(const QVector<float>& pred) {
    QVector<double> decoded;
    decoded.reserve(pred.size());
    for (float val : pred) {
        int idx = static_cast<int>(val + 0.5);
        if (idx >= 0 && idx < index_to_label_.size()) {
            decoded.append(index_to_label_[idx]);
        } else {
            decoded.append(-999); // or throw
        }
    }
    return decoded;
}

void XGBClassifier::Fit(const QVector<QVector<double>>& X,
                        const QVector<double>& y,
                        const QVector<float>& stabilizer,
                        float startProgressValue,
                        float endProgressValue) {
    QVector<float> y_encoded = EncodeLabels(y);
    CreateDMatrix(X, dtrain_);

    safe_xgboost(XGDMatrixSetFloatInfo(dtrain_, "label", y_encoded.data(), y_encoded.size()));

    if (!stabilizer.isEmpty() && stabilizer.size() == y_encoded.size()) {
        QVector<float> sample_weights(y_encoded.size(), 1.0f);
        for (int i = 0; i < stabilizer.size(); ++i) {
            float s = stabilizer[i];
            sample_weights[i] = 1.0f - 0.9f * s;
            if (sample_weights[i] < 0.01f)
                sample_weights[i] = 0.01f;
        }
        float total_weight = std::accumulate(sample_weights.begin(), sample_weights.end(), 0.0f);
        float mean_weight = total_weight / sample_weights.size();
        for (float& w : sample_weights) w /= mean_weight;

        safe_xgboost(XGDMatrixSetFloatInfo(dtrain_, "weight", sample_weights.data(), sample_weights.size()));
    }

    safe_xgboost(XGBoosterCreate(&dtrain_, 1, &booster_));

    int num_class = index_to_label_.size();
    params_["num_class"] = QString::number(num_class);
    SetBoosterParams();

    int n_iter = params_.contains("num_boost_round")
        ? params_["num_boost_round"].toInt()
        : 10;

    float progressWidth = endProgressValue - startProgressValue;

    for (int i = 0; i < n_iter; ++i) {
        if (terminated_) {
            qWarning("Training was terminated by user.");
            return;
        }
        safe_xgboost(XGBoosterUpdateOneIter(booster_, i, dtrain_));
        emit progress(startProgressValue + progressWidth * float(i + 1) / n_iter);
    }
}

void XGBClassifier::Fit(const QVector<QVector<double>>& X,
                        const QVector<double>& y,
                        float startProgressValue,
                        float endProgressValue) {
    QVector<float> empty_stabilizer;
    Fit(X, y, empty_stabilizer, startProgressValue, endProgressValue);
}



QVector<double> XGBClassifier::Predict(const QVector<QVector<double>>& X) {
    DMatrixHandle dtest;
    CreateDMatrix(X, dtest);

    bst_ulong out_len;
    const float* out_result;
    safe_xgboost(XGBoosterPredict(booster_, dtest, 0, 0, 0, &out_len, &out_result));

    QVector<float> raw;
    raw.reserve(out_len);
    for (bst_ulong i = 0; i < out_len; ++i)
        raw.append(out_result[i]);

    QVector<double> result = DecodeLabels(raw);
    XGDMatrixFree(dtest);
    return result;
}
